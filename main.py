from game import Game
from game_AI import Game_AI, Bird_AI

import sys


def main():
    if "-play" in sys.argv:
        game = Game()
        game.play()

    elif "-ai" in sys.argv:
        game = Game_AI()

        if "-bm" in sys.argv:
            bird = Bird_AI()
            bird.load_model("models/best_100.pkl")
            game.play_AI(bird)
        elif "-train" in sys.argv:
            game.train(n_generations=30, n_birds=30, limit_score=100, draw=False)
        else:
            game.play_AI()


if __name__ == '__main__':
    main()