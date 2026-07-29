import torch
from torch import nn
from typing import Optional
import transformers
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList


# --------------------------------------------------------------------------- #
#  Patched _sample with APC (Adaptive Plausibility Constraint)
#  Compatible with transformers >= 4.36 unified _sample signature.
#
#  Mechanism:
#    - A second "contrastive" forward pass uses the SAME inputs as the main
#      pass (no image distortion), giving logits_cd ≈ logits.
#    - cd_beta controls the APC cutoff:
#        cutoff = log(beta) + max_logit
#        tokens with logit < cutoff are masked to -inf
#    - beta=0  → cutoff=-inf → no masking → equivalent to direct sampling
#    - beta→1  → cutoff=max  → only argmax survives → equivalent to greedy
# --------------------------------------------------------------------------- #

def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config,
    synced_gpus: bool = False,
    streamer=None,
    **model_kwargs,
):
    # --- extract generation config ---
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(c, "eos_token_id") for c in stopping_criteria)
    do_sample = generation_config.do_sample

    # --- output accumulators ---
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None

    # --- init loop state ---
    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

    # APC: copy model_kwargs for the contrastive branch (same inputs, same image)
    model_kwargs_cd = model_kwargs.copy()
    use_apc = model_kwargs.get("use_apc") is not None

    # --- generation loop ---
    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):

        # main forward
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :].to(
            copy=True, dtype=torch.float32, device=input_ids.device
        )

        if use_apc:
            # APC: contrastive forward with same inputs (no image distortion)
            model_inputs_cd = self.prepare_inputs_for_generation(input_ids, **model_kwargs_cd)
            outputs_cd = self(
                **model_inputs_cd,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
            )
            next_token_logits_cd = outputs_cd.logits[:, -1, :].to(
                copy=True, dtype=torch.float32, device=input_ids.device
            )

            cd_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 1.0
            cd_beta  = model_kwargs.get("cd_beta")  if model_kwargs.get("cd_beta")  is not None else 0.2

            cutoff = (
                torch.log(torch.tensor(cd_beta, dtype=torch.float32, device=input_ids.device))
                + next_token_logits.max(dim=-1, keepdim=True).values
            )
            diffs     = (1 + cd_alpha) * next_token_logits - cd_alpha * next_token_logits_cd
            cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            next_token_scores = logits_processor(input_ids, cd_logits)
            del outputs_cd
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)

        # store optional outputs
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            from transformers.generation.utils import GenerateEncoderDecoderOutput
            return GenerateEncoderDecoderOutput(
                sequences=input_ids, scores=scores,
                encoder_attentions=encoder_attentions, encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions, cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            from transformers.generation.utils import GenerateDecoderOnlyOutput
            return GenerateDecoderOnlyOutput(
                sequences=input_ids, scores=scores,
                attentions=decoder_attentions, hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    return input_ids


def evolve_apc_sampling():
    transformers.generation.utils.GenerationMixin._sample = sample
