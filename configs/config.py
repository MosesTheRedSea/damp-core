
BASE = "/home/moses/Moses/Research/Current/Institute of Science Tokyo/acoustic-robotics"
PROJECT = "Multi-Task Acoustic Perception for Occluded Object Detection, Distance Estimation, and Material Classification"
LOSS = 1e-4 
EPOCHS = 100

OBJECT_TO_MATERIAL = {

        "no_object": "none", # completed

        # fabric
        "bookbag": "fabric",
        "rags": "fabric",
        "hoodie": "fabric",

        # paper_cardboard
        "textbooks": "paper_cardboard", # complete
        "paper": "paper_cardboard", # complete
        "cardboard_box": "paper_cardboard", # complete
        "bardboard_sheet": "paper_cardboard",

        # plastic
        "plastic_bottle": "plastic", # complete
        "container": "plastic", # complete
        "trash_bin": "plastic",
        "plastic_cup": "plastic", # complete
        "plastic_bowl": "plastic", # Complete
        "speaker": "plastic", # completed

        # metal
        "pot": "metal",  # completed
        "strainer": "metal", # completed
        "pitcher": "metal", # completed
        "sign": "metal", 
        "ladder": "metal", # completed
        "rolling_chair": "metal",

        # glass
        "glass_bottle": "glass", # complete
        "glass_cup": "glass", # complete
        "glass_bowl": "glass",
        "glass_mup": "glass",

        # ceramic
        "ceramic_mug": "ceramic", # complete
        "ceramic_bowl": "ceramic", #complete
        "plate": "ceramic", # complete

        # sand
        "sandbag": "sand"
    }

OBJ_CLASSES = [

    "no_object",

    # fabric
    "bookbag",
    "rags",

    # wood

    # paper_cardboard
    "textbooks",
    "paper",
    "cardboard_box",

    # plastic
    "plastic_bottle",
    "container",
    "trash_bin",
    "plastic_cup",
    "plastic_bowl",

    # metal
    "pot",
    "pitcher",
    "strainer",
    "sign",
    "ladder",
    "rolling_chair",

    # glass
    "glass_bottle",
    "glass_cup",
    "glass_bowl"

    # ceramic
    "mug",
    "ceramic_bowl",
    "plate",

    #sand
    "sandbag"
]

# 9 Materials
MAT_CLASSES = [
    "none",
    "wood",
    "metal",
    "plastic",
    "fabric",
    "glass",
    "ceramic",
    "paper_cardboard",
    "sand",
]