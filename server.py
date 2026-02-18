from openreward.environments import Server

from chinesimpleqa import ChineseSimpleQA

if __name__ == "__main__":
    Server([ChineseSimpleQA]).run()
