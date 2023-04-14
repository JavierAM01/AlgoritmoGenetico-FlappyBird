from game import Game
from game_AI import Game_AI


def main():
    game = Game_AI()
    game.train(n_generations=10, n_birds=10)


if __name__ == '__main__':
    main()