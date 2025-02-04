from src.script import main as main_script
from src.ai import main as main_ai_script

use_AI = True


def main(use_AI):
    if use_AI:
        main_ai_script()
    else:
        main_script()


if __name__ == "__main__":
    main()
