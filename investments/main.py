from models import main as run_models
from preprocess import main as run_preprocess


def main():
    run_preprocess()
    run_models()


if __name__ == "__main__":
    main()
