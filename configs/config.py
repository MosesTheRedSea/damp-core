
BASE = "/home/moses/Moses/Research/Current/Institute of Science Tokyo/acoustic-robotics"
PROJECT = "Multi-Task Acoustic Perception for Occluded Object Detection, Distance Estimation, and Material Classification"
LOSS = 1e-4 
EPOCHS = 100

OBJECT_TO_MATERIAL = {

        # make sure to include orientation of the items that you record 
        # also edit orientation of the chair

        # focus on getting sim2 real down to generate data, train the simulator
        # use the rover to test whether the data I record, can it acatually estimate the surrounding objects using the model I trained from the simulator.
        # if those recordings can be used to train the model. Simulator does the bulk of the training.
        # other data from the other room I can use it for testing
        # take my measurement to the other room and see whether it works

        # Sim2Real - MAINLY IMPORTANT TO GENERATE DATA 

        "no_object": "none", # completed 

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
        "rolling_chair": "metal", # make sure to record this
        "metal_cup":"cup",

        # glass
        "glass_vodka": "glass", # complete
        "glass_shooter": "glass",
        "glass_mug": "glass",

        # ceramic
        "ceramic_mug": "ceramic", # complete
        "ceramic_bowl": "ceramic", #complete
        "plate": "ceramic", # complete
        "teapot": "ceramic",
    }

OBJ_CLASSES = [

    "no_object",

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
    "plate"
    "helmet"
   # "monitor" # record this
]

# 9 Materials
MAT_CLASSES = [
    "none",
    "wood",
    "metal",
    "plastic",
    "glass",
    "ceramic",
    "paper_cardboard",
]