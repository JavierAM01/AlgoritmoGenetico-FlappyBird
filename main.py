from game import Game
from game_AI import Game_AI, Bird_AI

import sys


def main():
    if "-play" in sys.argv:
        game = Game()
        game.play()

    elif "-ai" in sys.argv:
        game = Game_AI()

        if "-train" in sys.argv:
            game.train(n_generations=30, n_birds=30, limit_score=100, draw=False)        
        else: # play with the best model
            bird = Bird_AI()
            bird.load_model("models/best_100.pkl")
            game.play_AI(bird)
    else:
        print("Para poder jugar debes poner alguno de los siguientes argumentos:")
        print(" -play : para jugar al juego como usuario.")
        print(" -ai : para ver jugar a la ia.")
        print(" -ai -train : para realizar un entrenamiento.")


if __name__ == '__main__':
    main()
