

PROJECT = "/home/3/um07293/research/occul-net"

LOSS = 1e-4 
EPOCHS = 100
FORCE_REGENERATE = False

TRAIN_MATERIAL = {  
    "none",
    "metal",
    "plastic",
    "paper_cardboard",
    "sand"
}

TRAIN_OBJECT = {

    "no_object",

    # paper_cardboard
    "cardboard_box",

    # plastic
    "speaker",

    # metal
    "pot",
    "pitcher",
    "strainer",
    "ladder",

    #sand
    "sandbag"
}

TRAIN_OBJECT_TO_MATERIAL = {

    "no_object": "none", # completed

    # paper_cardboard
    "cardboard_box": "paper_cardboard", # complete

    # plastic
    "speaker": "plastic", # completed

    # metal
    "pot": "metal",  # completed
    "strainer": "metal", # completed
    "pitcher": "metal", # completed
    "ladder": "metal", # completed

    "sandbag": "sand"
}


OBJECT_TO_MATERIAL = {

        "no_object": "none", # completed

        # fabric
        "bookbag": "fabric",
        "rags": "fabric",

        # paper_cardboard
        "textbooks": "paper_cardboard", # complete
        "paper": "paper_cardboard", # complete
        "cardboard_box": "paper_cardboard", # complete

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

        # ceramic
        "mug": "ceramic", # complete
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
    ""

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