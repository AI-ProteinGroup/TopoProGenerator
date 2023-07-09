import argparse
import json
import os

parser = argparse.ArgumentParser(description="you should add those parameter for pretrain")
parser.add_argument("--config", dest="config", help="path to the JSON config file", required=True)
args = parser.parse_args()

with open(args.config) as config_file:
    config = json.load(config_file)

print(config)
