class Config:
    NEXT_X = [0, 0,  1, -1]
    NEXT_Y = [1, -1, 0, 0]

    ACTION_SPACE = 4

    PREFERENCE_NUM = 10 ### 将90°分成10份 ###
    PREFERENCE_DIM = 2
    H,W = 15, 15
    TREASURE_COUNT = 10
    MAP_DIM = TREASURE_COUNT * 3

    LAMDA_COUNT = 500000  ### lamda 从 0 线性增长到 1 ###
    LAMDA_STEP = LAMDA_COUNT / 5

    BATCH_SIZE = 128
    SAMPLE_PREFERENCE = PREFERENCE_NUM // 2 ### 随机取出一半 ###

    ### st at r1 r2 st+1 done curw1 curw2 ###
    DATA_SHAPES = [MAP_DIM, 1, 1, 1, MAP_DIM, 1, 1, 1]

    LEARNING_RATE = 0.0001

    MULTIP_COUNT = 32  ### 8个policy网络 ###


