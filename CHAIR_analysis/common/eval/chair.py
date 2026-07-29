"""CHAIR: Caption Hallucination Assessment with Image Relevance.

Implementation of the object-hallucination metrics from:

    Rohrbach, Hendricks, Burns, Darrell, Saenko.
    "Object Hallucination in Image Captioning." EMNLP 2018.
    https://aclanthology.org/D18-1437.pdf

Two metrics are reported (both lower-is-better):

  * **CHAIR_i** (per-instance) = (# hallucinated MSCOCO objects mentioned)
                                / (# MSCOCO objects mentioned)
  * **CHAIR_s** (per-sentence) = (# captions with >= 1 hallucinated object)
                                / (# captions)

An object word in a generated caption is counted as *hallucinated* when it is
not part of the image's ground-truth object set.  The ground-truth set for an
image is the union of:

  1. objects present in the COCO instance segmentation annotations, and
  2. objects mentioned in the 5 human reference captions

exactly as in the original CHAIR code (this double source reduces false
hallucinations caused by COCO's incomplete segmentation labelling).

This module is dependency-light: it only needs the standard library.  Caption
tokenisation uses a simple ``\\w+`` regulariser and a hand-written singulariser
instead of the original code's ``nltk`` + ``pattern.en`` dependencies, which
matches the original behaviour for the object vocabulary used by CHAIR.
"""

import json
import os
import re


# ---------------------------------------------------------------------------
# MSCOCO object synonyms.
#
# One line per MSCOCO category.  The FIRST token on each line is the canonical
# category name (the "node word"); the remaining tokens are synonyms that map
# back to it.  This is the canonical ``synonyms.txt`` from the CHAIR release
# (https://github.com/LisaAnne/Hallucination), embedded here so no external
# file is required.
# ---------------------------------------------------------------------------
SYNONYMS_TXT = """\
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, foreigner, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycles, biker, motorcyclist
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter, moped
airplane, jet, plane, jetliner, airbus, aircraft, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, yacht, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, speedboat, canoe, gondola, kayak, fishingboat
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, goat, ewe
cow, cattle, oxen, ox, calf, holstein, heifer, buffalo, bull, zebu, bison
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bowtie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle, flask
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake, cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iphone
microwave
oven, stovetop, stove, stoves
toaster
sink
refrigerator, fridge, fridges, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
"""


_WORD_RE = re.compile(r"[a-z]+")


def _tokenize(text):
    """Lower-case and split a caption into alphabetic word tokens."""
    return _WORD_RE.findall(text.lower())


def singularize(word):
    """Lightweight English singulariser sufficient for the CHAIR vocabulary.

    Mirrors the intent of ``pattern.en.singularize`` for the object nouns CHAIR
    cares about (e.g. ``chairs`` -> ``chair``, ``buses`` -> ``bus``,
    ``knives`` -> ``knife``, ``berries`` -> ``berry``) without pulling in the
    ``pattern`` dependency.
    """
    irregular = {
        "men": "man",
        "women": "woman",
        "people": "people",
        "children": "child",
        "geese": "goose",
        "feet": "foot",
        "teeth": "tooth",
        "mice": "mouse",
        "buses": "bus",
        "skis": "skis",
        "knives": "knife",
        "scissors": "scissors",
        "glasses": "glass",
    }
    if word in irregular:
        return irregular[word]
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ves"):
        return word[:-3] + "f"
    if word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if word.endswith("oes"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


class CHAIR(object):
    """Compute CHAIR metrics for a set of image ids against COCO annotations."""

    def __init__(self, imids, coco_path):
        """``imids`` is the iterable of COCO image ids being evaluated.

        ``coco_path`` is the directory holding ``instances_val2017.json`` and
        ``captions_val2017.json`` (i.e. ``data/coco/annotations``).
        """
        self.imid_to_objects = {imid: set() for imid in imids}
        self.coco_path = coco_path

        synonyms = [
            [tok.strip() for tok in line.split(",")]
            for line in SYNONYMS_TXT.strip().splitlines()
        ]
        self.mscoco_objects = []  # canonical names + every synonym
        self.inverse_synonym_dict = {}  # any synonym -> canonical name
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]
        self.mscoco_objects = set(self.mscoco_objects)

        # 'double words' that should be collapsed to a single token before the
        # vocabulary lookup (otherwise e.g. "baseball bat" would be missed and
        # "bat" alone is not a COCO object).
        coco_double_words = [
            "motor bike", "motor cycle", "air plane", "traffic light",
            "street light", "traffic signal", "stop light", "fire hydrant",
            "stop sign", "parking meter", "suit case", "sports ball",
            "baseball bat", "baseball glove", "tennis racket", "wine glass",
            "hot dog", "cell phone", "mobile phone", "teddy bear", "hair drier",
            "potted plant", "bow tie", "laptop computer", "stove top oven",
            "home plate", "train track",
        ]
        animal_words = [
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
            "zebra", "giraffe", "animal", "cub",
        ]
        vehicle_words = ["jet", "train"]

        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict["baby %s" % animal_word] = animal_word
            self.double_word_dict["adult %s" % animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict["passenger %s" % vehicle_word] = vehicle_word
        self.double_word_dict["bow tie"] = "tie"
        self.double_word_dict["toilet seat"] = "toilet"
        self.double_word_dict["wine glas"] = "wine glass"

    def caption_to_words(self, caption):
        """Map a caption to the MSCOCO objects it mentions.

        Returns ``(words, node_words, idxs, double_words)`` where ``node_words``
        are the canonical category names mentioned (with duplicates preserved).
        """
        words = [singularize(w) for w in _tokenize(caption)]

        # collapse double words
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
            idxs.append(i)
            double_word = " ".join(words[i:i + 2])
            if double_word in self.double_word_dict:
                double_words.append(self.double_word_dict[double_word])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words

        # "the seat of the toilet" should not fire the chair/seat synonym
        if ("toilet" in words) and ("seat" in words):
            words = [w for w in words if w != "seat"]

        idxs = [idxs[idx] for idx, w in enumerate(words) if w in self.mscoco_objects]
        words = [w for w in words if w in self.mscoco_objects]
        node_words = [self.inverse_synonym_dict[w] for w in words]
        return words, node_words, idxs, double_words

    # -- ground-truth assembly -------------------------------------------------

    def _get_annotations_from_segments(self, instances_file):
        coco_segments = json.load(open(instances_file))
        id_to_name = {c["id"]: c["name"] for c in coco_segments["categories"]}
        for ann in coco_segments["annotations"]:
            imid = ann["image_id"]
            if imid in self.imid_to_objects:
                name = id_to_name[ann["category_id"]]
                node_word = self.inverse_synonym_dict[name]
                self.imid_to_objects[imid].add(node_word)

    def _get_annotations_from_captions(self, captions_file):
        coco_caps = json.load(open(captions_file))
        for ann in coco_caps["annotations"]:
            imid = ann["image_id"]
            if imid in self.imid_to_objects:
                _, node_words, _, _ = self.caption_to_words(ann["caption"])
                self.imid_to_objects[imid].update(node_words)

    def get_annotations(self, instances_file=None, captions_file=None):
        """Populate ``imid_to_objects`` from COCO instances + reference captions."""
        if instances_file is None:
            instances_file = os.path.join(self.coco_path, "instances_val2017.json")
        if captions_file is None:
            captions_file = os.path.join(self.coco_path, "captions_val2017.json")
        self._get_annotations_from_segments(instances_file)
        self._get_annotations_from_captions(captions_file)

    # -- scoring ---------------------------------------------------------------

    def compute_chair(self, caps):
        """Compute CHAIR over ``caps`` = list of {"image_id", "caption"} dicts.

        Returns a results dict with the aggregate metrics and per-sentence
        breakdown (hallucinated objects per caption).
        """
        num_caps = 0.0
        num_hallucinated_caps = 0.0
        hallucinated_word_count = 0.0
        coco_word_count = 0.0
        total_len = 0.0

        sentences = []
        for cap in caps:
            imid = cap["image_id"]
            if imid not in self.imid_to_objects:
                continue
            cap_text = cap["caption"]
            words, node_words, idxs, raw_words = self.caption_to_words(cap_text)
            gt_objects = self.imid_to_objects[imid]

            hallucinated = []
            for word, node_word, idx in zip(words, node_words, idxs):
                if node_word not in gt_objects:
                    hallucinated.append(node_word)

            coco_word_count += len(node_words)
            hallucinated_word_count += len(hallucinated)
            if len(hallucinated) > 0:
                num_hallucinated_caps += 1
            num_caps += 1
            total_len += len(raw_words)

            sentences.append({
                "image_id": imid,
                "caption": cap_text,
                "mscoco_objects": list(set(node_words)),
                "hallucinated_objects": list(set(hallucinated)),
                "gt_objects": list(gt_objects),
            })

        chair_s = num_hallucinated_caps / num_caps if num_caps else 0.0
        chair_i = hallucinated_word_count / coco_word_count if coco_word_count else 0.0

        return {
            "num_captions": int(num_caps),
            "CHAIR_s": chair_s,
            "CHAIR_i": chair_i,
            "avg_objects_per_caption": coco_word_count / num_caps if num_caps else 0.0,
            "avg_caption_length": total_len / num_caps if num_caps else 0.0,
            "sentences": sentences,
        }
