import logging
import os
from typing import Dict, Any
from datetime import datetime

from my_proof.proof_of_ownership import verify_ownership
from my_proof.proof_of_quality_n_authenticity import process_files_for_quality_n_authenticity_scores
from my_proof.models.proof_response import ProofResponse
from my_proof.proof_of_uniqueness import process_files_for_uniqueness

# Ensure logging is configured
logging.basicConfig(level=logging.INFO)

CONTRIBUTION_THRESHOLD = 4
EXTRA_POINTS = 5

TOKEN_MAPPING = {
    "browser_history": int(os.environ.get("BROWSER_HISTORY_TOKEN_COUNT", 15)),  # maps to csv
    "bookmark_history": int(os.environ.get("BOOKMARK_HISTORY_TOKEN_COUNT", 10)),  # maps to html/yaml
    "location_timeline": int(os.environ.get("LOCATION_TIMELINE_TOKEN_COUNT", 25))  # maps to json
}

class Proof:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.proof_response = ProofResponse(dlp_id=config['dlp_id'])
        self.wallet_address = ""
    
    def read_author_from_file(self, file_path: str):
        """
        Read parameters from a text file.

        :param file_path: Path to the text file
        :return: Tuple containing author, signature, and random_string
        """
        params = {}
        with open(file_path, "r") as file:
            for line in file:
                key, value = line.strip().split(": ", 1)
                params[key] = value
        return params["author"]

    def calculate_total_tokens(self, platform_rewards):
        return sum(reward["token_reward"] for reward in platform_rewards.values())

    def calculate_final_score(self, total_tokens):
        max_possible_tokens = sum(TOKEN_MAPPING.values())
        return total_tokens / max_possible_tokens if max_possible_tokens > 0 else 0.0

    def generate(self) -> ProofResponse:
        """Generate proofs for all input files."""
        logging.info("Starting proof generation")

        proof_response_object = {
            'dlp_id': self.config.get('dlp_id', 30),
            'valid': True,
        }

        txt_files = [f for f in os.listdir(self.config['input_dir']) if f.endswith('.txt')]
        if txt_files:
            self.wallet_address = self.read_author_from_file(os.path.join(self.config['input_dir'], txt_files[0])).lower()
            logging.info(f"Wallet Address {self.wallet_address}")

        platform_rewards = {}
        platform_types = []

        for input_filename in os.listdir(self.config['input_dir']):
            logging.info(f"Processing file: {input_filename}")
            file_id = self.config.get('file_id')
            logging.info(f"Processing file ID: {file_id}")

            # Process uniqueness
            uniqueness_details = process_files_for_uniqueness(file_id, self.config['input_dir'], self.wallet_address)
            data_types_provided = uniqueness_details.get("data_types_provided", [])

            # Process quality & authenticity
            quality_n_authenticity_details = process_files_for_quality_n_authenticity_scores(
                uniqueness_details.get("unique_csv_data"),
                uniqueness_details.get("unique_json_data"),
                uniqueness_details.get("unique_yaml_entries")
            )

            proof_response_object['ownership'] = verify_ownership(self.config['input_dir'])
            proof_response_object['uniqueness'] = uniqueness_details.get("uniqueness_score")
            proof_response_object['quality'] = quality_n_authenticity_details.get("quality_score")
            proof_response_object['authenticity'] = quality_n_authenticity_details.get("authenticity_score")

            if proof_response_object['authenticity'] < 1.0:
                proof_response_object['valid'] = True

        mapped_types = set()
        for data_type in data_types_provided:
            key_mapping = {"csv": "browser_history", "json": "location_timeline", "yaml": "bookmark_history"}
            key = key_mapping.get(data_type)
            if not key:
                continue

            mapped_types.add(key)
            uniqueness_percentage = uniqueness_details.get(f"{data_type}_uniqueness_score", 0.0)
            token_reward = TOKEN_MAPPING[key] * uniqueness_percentage

            scores = {
                "uniqueness": uniqueness_details.get(f"{data_type}_uniqueness_score", 0.0),
                "quality": quality_n_authenticity_details.get(f"{data_type}_quality_score", 0.0),
                "authenticity": quality_n_authenticity_details.get(f"{data_type}_authenticity_score", 0.0),
                "ownership": proof_response_object['ownership']
            }
            platform_rewards[key] = {
                "token_reward": token_reward,
                **scores,
                "score": sum(scores.values()) / len(scores)
            }

        total_tokens = self.calculate_total_tokens(platform_rewards)
        proof_response_object["metadata"] = {
            "submission_time": datetime.now().isoformat(),
            "total_tokens": total_tokens,
            "types": list(mapped_types),
            "platform_rewards": platform_rewards
        }

        proof_response_object['score'] = self.calculate_final_score(total_tokens)
        logging.info(f"Proof response: {proof_response_object}")
        return proof_response_object
