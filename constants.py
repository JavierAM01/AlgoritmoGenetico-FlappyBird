import pygame, os


def load_image(name, size = False):
    image = pygame.image.load(os.path.join("images", name))
    if size != False: image = pygame.transform.scale(image, size)
    return image


# sizes
SIZE_window    = (400, 700)
SIZE_floor     = (400, 110)
SIZE_pipe      = (70, 400)
SIZE_number    = (40, 50)
SIZE_message   = (300, 500)
SIZE_game_over = (300, 100)


# background images
bg_type = "day" # "night"
IMG_backgroung = load_image("background-" + bg_type + ".png", SIZE_window)
IMG_floor      = load_image("base.png", SIZE_floor)
IMG_pipe_up    = load_image("pipe-green.png", SIZE_pipe)
IMG_pipe_down  = pygame.transform.flip(IMG_pipe_up, False, True)
YLIM = 600 # bottom limit for the bird (when the floor starts)


# numbers images
IMG_numbers = {}
for i in range(10): IMG_numbers[i] = load_image(str(i) + ".png", SIZE_number)


# bird images
size_bird = (45, 38)
IMG_bird_mid_red     = load_image("redbird-midflap.png", size_bird)
IMG_bird_down_red    = load_image("redbird-downflap.png", size_bird)
IMG_bird_up_red      = load_image("redbird-upflap.png", size_bird)
IMG_bird_mid_yellow  = load_image("yellowbird-midflap.png", size_bird)
IMG_bird_down_yellow = load_image("yellowbird-downflap.png", size_bird)
IMG_bird_up_yellow   = load_image("yellowbird-upflap.png", size_bird)
IMG_bird_mid_blue    = load_image("bluebird-midflap.png", size_bird)
IMG_bird_down_blue   = load_image("bluebird-downflap.png", size_bird)
IMG_bird_up_blue     = load_image("bluebird-upflap.png", size_bird)

IMG_birds = dict()
IMG_birds["blue"]   = [pygame.transform.rotate(IMG_bird_up_blue, 5),   IMG_bird_mid_blue,   pygame.transform.rotate(IMG_bird_down_blue, -5)  ]
IMG_birds["red"]    = [pygame.transform.rotate(IMG_bird_up_red, 5),    IMG_bird_mid_red,    pygame.transform.rotate(IMG_bird_down_red, -5)   ]
IMG_birds["yellow"] = [pygame.transform.rotate(IMG_bird_up_yellow, 5), IMG_bird_mid_yellow, pygame.transform.rotate(IMG_bird_down_yellow, -5)]


# extra info
IMG_mesage    = load_image("message.png", SIZE_message)
IMG_game_over = load_image("gameover.png", SIZE_game_over)

