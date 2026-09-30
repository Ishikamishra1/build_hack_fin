from handler import lambda_handler
import json
result = lambda_handler({"therapeutic_area": "colorectal cancer", "max_results": 5}, None)
print(json.dumps(result, indent=2))

