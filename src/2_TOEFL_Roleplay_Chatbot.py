#!/usr/bin/env python
# coding: utf-8

# # TOEFL Roleplay Test (Back-and-Forth) + Auto Grading (Gemini)
# 
# This notebook runs a short TOEFL-style roleplay "test session" and returns a structured grade at the end.
# 
# ### Why back-and-forth?
# In real speaking tests, performance is shown across multiple turns (clarifying, responding, negotiating meaning). A multi-turn roleplay is closer to real test behavior than a single prompt.
# 
# ### Why structured JSON output?
# We want consistent results for dashboards and product demos. JSON lets us store scores, comments, and follow-up prompts in a predictable format.
# 
# ### Design goals
# - Test-like: neutral evaluator tone, no coaching during the test
# - Short messages (good for demos)
# - Reliable parsing (avoid "half JSON" failures)
# 

# In[1]:


from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError
from google import genai
from google.genai import types

# # 1) API Key setup 

# In[2]:


api_key= os.environ["GEMINI_API_KEY"] = "GEMINI_API_KEY"
client = genai.Client(api_key=api_key)

# In[3]:


client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"

# ## Key parameters (what they control)
# 
# ### `max_turns`
# How many candidate replies happen before we end the test and grade.  
# - Smaller (3–4) is best for quick client demos.
# 
# ### `temperature`
# Controls randomness in the model's output.
# - Lower (0.0–0.2): more predictable, better for strict JSON.
# - Higher (0.5+): more creative, but increases formatting risk.
# 
# ### `max_output_tokens`
# Caps the length of the model output.
# - For examiner turns: keep low so questions stay short.
# - For final JSON: keep higher so the JSON doesn't get cut off mid-way.
# 
# ### Why we compact the transcript
# Long conversation history makes the model more likely to output incomplete JSON.
# So we keep only the last few turns before grading.
# 

# # 2) Pydantic schema

# In[4]:


class RubricCategory(BaseModel):
    score_0_5: float = Field(ge=0.0, le=5.0)
    comment: str = Field(max_length=140)


class RubricFeedback(BaseModel):
    task_fulfillment: RubricCategory
    coherence: RubricCategory
    language_use: RubricCategory


class TutorResponse(BaseModel):
    brief_summary: str = Field(max_length=200)  # max 2 sentences in prompt
    rubric_feedback: RubricFeedback
    top_fixes: List[str]  # enforce 3 items via prompt
    score_company: float = Field(ge=0.0, le=10.0)
    follow_up_prompt: str = Field(max_length=200)

# # 3) Scoring helper

# In[5]:


def compute_company_score(task: float, coherence: float, language: float) -> float:
    """
    Converts 0–5 rubric average into 0–10 company score.
    This keeps the scoring consistent regardless of model quirks.
    """
    avg_0_5 = (task + coherence + language) / 3.0
    return round(avg_0_5 * 2.0, 1)

# # 4) Scenario system (random scenarios each run)

# In[6]:


@dataclass(frozen=True)
class Scenario:
    title: str
    description: str


SCENARIOS: List[Scenario] = [
    Scenario(
        title="Faulty product return",
        description=(
            "TOEFL Roleplay Scenario:\n"
            "Situation: You bought a product online and it arrived faulty.\n"
            "Roles: EXAMINER is customer service; CANDIDATE is the customer.\n"
            "Constraints: Purchase was 12 days ago; refund over $50 needs approval; "
            "return label can be offered.\n"
        ),
    ),
    Scenario(
        title="Hotel reservation problem",
        description=(
            "TOEFL Roleplay Scenario:\n"
            "Situation: You booked a hotel room, but the room type is wrong on arrival.\n"
            "Roles: EXAMINER is front desk; CANDIDATE is the guest.\n"
            "Constraints: Limited rooms; offer upgrade/discount if needed.\n"
        ),
    ),
    Scenario(
        title="Late delivery complaint",
        description=(
            "TOEFL Roleplay Scenario:\n"
            "Situation: Your delivery is late and you need it urgently.\n"
            "Roles: EXAMINER is delivery support; CANDIDATE is the customer.\n"
            "Constraints: Apologize, give an update, offer reschedule/compensation.\n"
        ),
    ),
    Scenario(
        title="Refund for cancelled class",
        description=(
            "TOEFL Roleplay Scenario:\n"
            "Situation: A training class was cancelled and you want a refund.\n"
            "Roles: EXAMINER is admin; CANDIDATE is the student.\n"
            "Constraints: Refund depends on policy; offer reschedule.\n"
        ),
    ),
]


def pick_random_scenario(seed: int | None = None) -> Scenario:
    rng = random.Random(seed)
    return rng.choice(SCENARIOS)

# #  5) Conversation log + transcript compaction

# In[7]:


def start_test(conversation_log: List[str]) -> None:
    conversation_log.clear()


def add_examiner(conversation_log: List[str], text: str) -> None:
    conversation_log.append(f"EXAMINER: {text.strip()}")


def add_candidate(conversation_log: List[str], text: str) -> None:
    conversation_log.append(f"CANDIDATE: {text.strip()}")


def compact_conversation(
    conversation_log: List[str],
    max_lines: int = 14,
    max_chars: int = 2500,
) -> str:
    """
    Keeps only the latest part of the conversation.
    This reduces the chance of incomplete JSON on the final grade.
    """
    recent_lines = conversation_log[-max_lines:]
    text = "\n".join(recent_lines)
    return text[-max_chars:]

# # 6) Prompt builder (test mode + short outputs)

# In[8]:


def build_prompts(
    scenario_text: str,
    conversation_so_far: str,
    stage: str,
) -> Tuple[str, str]:
    system_instruction = (
        "You are a TOEFL evaluator in TEST MODE (not a tutor).\n"
        "Run a short roleplay speaking test.\n\n"
        "Rules:\n"
        "- Ask ONE question at a time.\n"
        "- Keep your messages short and complete.\n"
        "- Do NOT teach or correct during the test.\n"
        "- Only grade when asked at the end.\n\n"
        "Rubric (0–5 each): task_fulfillment, coherence, language_use.\n"
        "Company score (0–10): round(((task+coherence+language)/3)*2, 1)\n"
    )

    if stage == "continue":
        user_prompt = f"""
SCENARIO:
{scenario_text}

CONVERSATION SO FAR:
{conversation_so_far}

Continue the test:
- Ask the next ONE short examiner question.
- No grading. No JSON.
""".strip()
        return system_instruction, user_prompt

    if stage == "end":
     user_prompt = f"""
SCENARIO:
{scenario_text}

CONVERSATION SO FAR:
{conversation_so_far}

END THE TEST NOW.

Return ONLY one raw JSON object. No extra text. No markdown.

The JSON MUST follow this exact shape:

{{
  "brief_summary": "max 2 sentences",
  "rubric_feedback": {{
    "task_fulfillment": {{"score_0_5": 0.0, "comment": "max 1 sentence"}},
    "coherence": {{"score_0_5": 0.0, "comment": "max 1 sentence"}},
    "language_use": {{"score_0_5": 0.0, "comment": "max 1 sentence"}}
  }},
  "top_fixes": ["bullet 1", "bullet 2", "bullet 3"],
  "score_company": 0.0,
  "follow_up_prompt": "1 sentence"
}}

Rules:
- rubric_feedback values MUST be objects with score_0_5 and comment (not strings)
- top_fixes MUST have exactly 3 strings
""".strip()
    return system_instruction, user_prompt

# # 7) JSON parsing

# In[9]:


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    match = _JSON_BLOCK.search(text)
    return match.group(0).strip() if match else None


def parse_tutor_response(raw_text: str) -> TutorResponse:
    json_str = extract_json_object(raw_text)
    if json_str is None:
        raise ValueError("No JSON object found in model output.")

    data = json.loads(json_str)
    result = TutorResponse.model_validate(data)

    # Enforce consistent company scoring
    t = result.rubric_feedback.task_fulfillment.score_0_5
    c = result.rubric_feedback.coherence.score_0_5
    l = result.rubric_feedback.language_use.score_0_5
    result.score_company = compute_company_score(t, c, l)

    # This will keep it stable for any demos we show to the client
    result.top_fixes = [s.strip() for s in result.top_fixes][:3]
    return result

# # 8) Gemini calls: examiner turn + final grade

# In[10]:


def examiner_turn(
    scenario_text: str,
    conversation_log: List[str],
    model: str = MODEL_NAME,
) -> str:
    conversation_so_far = compact_conversation(conversation_log)
    system_instruction, user_prompt = build_prompts(
        scenario_text=scenario_text,
        conversation_so_far=conversation_so_far,
        stage="continue",
    )

    response = client.models.generate_content(
        model=model,
        contents=f"{system_instruction}\n\n{user_prompt}",
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=140,  # keep examiner questions short but complete
        ),
    )
    return (response.text or "").strip()


def final_grade(
    scenario_text: str,
    conversation_log: List[str],
    model: str = MODEL_NAME,
    retries: int = 2,
) -> Optional[TutorResponse]:
    conversation_so_far = compact_conversation(conversation_log)
    system_instruction, user_prompt = build_prompts(
        scenario_text=scenario_text,
        conversation_so_far=conversation_so_far,
        stage="end",
    )

    last_raw = ""
    last_error: Exception | None = None

    for attempt in range(1, retries + 2):
        response = client.models.generate_content(
            model=model,
            contents=f"{system_instruction}\n\n{user_prompt}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,        # stable JSON
                max_output_tokens=1200,  # avoid truncation
            ),
        )

        last_raw = response.text or ""
        try:
            return parse_tutor_response(last_raw)
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            print(f"Final grade parse attempt {attempt} failed: {exc}")
            print("RAW OUTPUT (first 220 chars):")
            print(last_raw[:220])
            print("-" * 60)

    print(" All attempts failed.")
    if last_error:
        print("Last error:", last_error)
    print("\nLast raw output:\n", last_raw)
    return None

# # 9) Automated session runner (For Demo)

# In[11]:


def build_demo_candidate_reply(turn: int, scenario_title: str) -> str:
    canned = {
        "Faulty product return": [
            "Hi, I received the item recently, but it doesn’t work at all.",
            "I charged it and followed the instructions, but nothing changed.",
            "I’d prefer a refund because I need a working item soon.",
            "Yes, I can return it. Please send the return label and next steps.",
        ],
        "Hotel reservation problem": [
            "Hi, I booked a queen room, but I was given a different room type.",
            "I’m travelling for work, so I really need the correct room tonight.",
            "If the correct room isn’t available, can you offer an upgrade or discount?",
            "Thanks. Please confirm the solution and any extra charges in writing.",
        ],
        "Late delivery complaint": [
            "Hi, my delivery is late and I need it urgently.",
            "I’ve been waiting all day and I need an update on the delivery status.",
            "If it can’t arrive today, I’d like a refund or compensation.",
            "Okay, please confirm the new delivery time and what you can offer.",
        ],
        "Refund for cancelled class": [
            "Hi, my class was cancelled and I’d like a refund.",
            "I already paid, and I’m not available to reschedule soon.",
            "Could you explain the refund timeline and method clearly?",
            "Thanks. Please email me confirmation of the refund request.",
        ],
    }
    lines = canned.get(scenario_title, [
        "Hi, I have an issue and I need help.",
        "Here are the details of the problem.",
        "I’d like a fair solution, please.",
        "Thank you. Please confirm the next steps.",
    ])
    return lines[min(turn - 1, len(lines) - 1)]


def run_test_session(
    max_turns: int = 4,
    seed: int | None = None,
    auto_demo: bool = True,
) -> Optional[TutorResponse]:
    """
    Runs one complete session:
    - picks a random scenario
    - runs max_turns of examiner Q + candidate A
    - ends with JSON grading

    auto_demo=True: uses canned candidate replies (best for client demos).
    auto_demo=False: asks you to type candidate replies.
    """
    scenario = pick_random_scenario(seed=seed)
    scenario_text = scenario.description

    conversation_log: List[str] = []
    start_test(conversation_log)

    add_examiner(
        conversation_log,
        "TOEFL test simulation. I will ask short questions. Answer naturally. Let’s begin."
    )

    print(f"\n Scenario: {scenario.title}\n")

    for turn in range(1, max_turns + 1):
        q = examiner_turn(scenario_text, conversation_log)
        add_examiner(conversation_log, q)
        print(f"EXAMINER (Q{turn}): {q}")

        if auto_demo:
            reply = build_demo_candidate_reply(turn, scenario.title)
            print(f"CANDIDATE: {reply}\n")
        else:
            reply = input("CANDIDATE: ").strip()

        add_candidate(conversation_log, reply)

    print("\n--- TRANSCRIPT ---")
    print("\n".join(conversation_log))

    print("\n--- FINAL GRADE ---")
    grade = final_grade(scenario_text, conversation_log)

    if grade is None:
        print("Grading failed (JSON formatting).")
    else:
        print(grade.model_dump())

    return grade

# # 10) Running it, Random scenario each time

# In[13]:


run_test_session(max_turns=3, auto_demo=True, seed=None)

# In[ ]:



