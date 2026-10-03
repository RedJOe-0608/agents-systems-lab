import hashlib


def count_r(text):
    """Count occurrences of the letter r, ignoring case."""
    return text.lower().count("r")


def calculate_sha256(text):
    """Calculate the SHA-256 hash of a string."""
    encoded_text = text.encode("utf-8")
    return hashlib.sha256(encoded_text).hexdigest()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "count_r",
            "description": (
                "Count occurrences of the letter r, ignoring case, "
                "in a string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The exact text to inspect.",
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_sha256",
            "description": (
                "Calculate the exact SHA-256 hash of a string "
                "and return its hexadecimal digest."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The exact text to hash.",
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
]


AVAILABLE_TOOLS = {
    "count_r": count_r,
    "calculate_sha256": calculate_sha256,
}