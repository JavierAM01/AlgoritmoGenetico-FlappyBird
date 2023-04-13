from env import Enviroment
from model import Net_FlappyBird as Net


def main():
    enviroment = Enviroment()

    """
    d = model.state_dict()
    print("model.state_dict():", d)
    for key in d:
        print("key:", key)
        print("d[key]:", d[key])
        break
    """
    
    enviroment.play()#AI_Gen(n_epochs=30, n_simulations=30)


if __name__ == '__main__':
    main()