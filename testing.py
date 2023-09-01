from langchain_llm_api.banana_llm import BananaLLM

from langchain.tools.json.tool import JsonSpec
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.agents import create_json_agent


PRO_data = {
  "title": "PeriodicRevisitObjective",
  "type": "object",
  "target_id": { "type": "integer" },
  "sensor_name": { "type": "string" },
  "data_mode": { "type": "string" },
  "classification_marking": { "type": "string", "default": "U" },
  "revisits_per_hour": { "type": "integer", "default": 1 },
  "hours_to_plan": { "type": "integer", "default": 24 },
  "objective_name": { "type": "string", "default": "Periodic Revisit Objective" },
  "objective_start_time": { "type": "string", "format": "date-time" },
  "objective_end_time": { "type": "string", "format": "date-time" },
  "priority": { "type": "integer", "default": 2 },
  "required": ["target_id", "sensor_name", "data_mode", "classification_marking"]
}

json_prompt = """Respond to the following in JSON with \'action\' and \'action_input\' values \nUser: Given the following user task: `Track object 12345 with sensor RME08, revisiting twice per hour for the next 16 hours`, Examine the following JSON schema:\n\n{\n  "title": "PeriodicRevisitObjective",\n  "type": "object",\n  "target_id": { "type": "integer" },\n  "sensor_name": { "type": "string" },\n  "data_mode": { "type": "string" },\n  "classification_marking": { "type": "string", "default": "U" },\n  "revisits_per_hour": { "type": "integer", "default": 1 },\n  "hours_to_plan": { "type": "integer", "default": 24 },\n  "objective_name": { "type": "string", "default": "Periodic Revisit Objective" },\n  "objective_start_time": { "type": "string", "format": "date-time" },\n  "objective_end_time": { "type": "string", "format": "date-time" },\n  "priority": { "type": "integer", "default": 2 },\n  "required": ["target_id", "sensor_name", "data_mode", "classification_marking"]\n}\n\n\nCreate a JSON object that represents an instance of the `PeriodicRevisitObjective` class using information from the user task. Here is an example for a JSON that adheres to the schema:\n\n[\'{\\n  "objective_def_name": "PeriodicRevisitObjective",\\n  "target_id": 28884,\\n  "sensor_name": "RME00",\\n  "revisits_per_hour": 3,\\n  "data_mode": "TEST",\\n  "classification_marking": "U",\\n  "objective_name": "Periodic Revisit Objective",\\n  }\\n\']\n\nCreate a new JSON object and include every parameter from the class if it is required. Required: target_id, sensor_name, data_mode, classification_marking from the JSON schema.\nIMPORTANT: Make sure to begin all your responses with \'action\' and \'action_input\' text [/INST]
"""

## From Banana dev (app.banana.dev)
api_key = ""
model_key = ""
url = ""

llm = BananaLLM(api_key, model_key, url)

json_spec = JsonSpec(dict_=PRO_data, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)

json_agent_executor = create_json_agent(
    llm=llm,
    toolkit=json_toolkit,
    verbose=True
    )

response = json_agent_executor.run(json_prompt)

print(response)
