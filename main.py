from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os, json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

# Correct argument order for each function
ARGUMENT_ORDER = {
    "get_ticket_status": ["ticket_id"],
    "schedule_meeting": ["date", "time", "meeting_room"],
    "get_expense_balance": ["employee_id"],
    "calculate_performance_bonus": ["employee_id", "current_year"],
    "report_office_issue": ["issue_code", "department"],
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_ticket_status",
            "description": "Get the status of an IT support ticket",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "integer"}
                },
                "required": ["ticket_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_meeting",
            "description": "Schedule a meeting on a date, time and room",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "time": {"type": "string"},
                    "meeting_room": {"type": "string"}
                },
                "required": ["date", "time", "meeting_room"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_expense_balance",
            "description": "Get expense reimbursement balance for an employee",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "integer"}
                },
                "required": ["employee_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_performance_bonus",
            "description": "Calculate performance bonus for an employee for a given year",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "integer"},
                    "current_year": {"type": "integer"}
                },
                "required": ["employee_id", "current_year"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "report_office_issue",
            "description": "Report an office issue with issue code and department",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_code": {"type": "integer"},
                    "department": {"type": "string"}
                },
                "required": ["issue_code", "department"]
            }
        }
    }
]

@app.get("/execute")
async def execute(q: str):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts function calls from user queries. Always call the most appropriate function."},
                {"role": "user", "content": q}
            ],
            tools=tools,
            tool_choice="required"
        )

        tool_call = response.choices[0].message.tool_calls[0]
        func_name = tool_call.function.name
        raw_args = json.loads(tool_call.function.arguments)

        # Re-order arguments to match function definition order
        ordered_args = {}
        for key in ARGUMENT_ORDER.get(func_name, raw_args.keys()):
            if key in raw_args:
                ordered_args[key] = raw_args[key]

        return {
            "name": func_name,
            "arguments": json.dumps(ordered_args)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
