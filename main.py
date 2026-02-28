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

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_ticket_status",
            "description": "Get the status of an IT support ticket",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "integer", "description": "The ticket ID number"}
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
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                    "time": {"type": "string", "description": "Time in HH:MM format"},
                    "meeting_room": {"type": "string", "description": "The meeting room name"}
                },
                "required": ["date", "time", "meeting_room"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_expense_balance",
            "description": "Get the expense reimbursement balance for an employee",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "integer", "description": "The employee ID number"}
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
                    "employee_id": {"type": "integer", "description": "The employee ID number"},
                    "current_year": {"type": "integer", "description": "The year for bonus calculation"}
                },
                "required": ["employee_id", "current_year"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "report_office_issue",
            "description": "Report an office issue with an issue code and department",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_code": {"type": "integer", "description": "The issue code number"},
                    "department": {"type": "string", "description": "The department name"}
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
        return {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
