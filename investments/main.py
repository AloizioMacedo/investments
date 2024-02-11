from models import main as run_models
from outputs import main as run_outputs
from preprocess import main as run_preprocess


def main():
    run_preprocess()
    run_models()
    run_outputs()


if __name__ == "__main__":
    main()
