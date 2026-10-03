from agent_harness.compaction import compact_old_messages
from agent_harness.runtime import run_agent_turn

MAX_STEPS = 5
MAX_CONTEXT_TURNS = 3 # decides how many recent turns stay in the context verbatim. The older turns become eligible for compaction.
COMPACTION_BATCH_TURNS = 3 # decides when to trigger compaction on the older elegible turns.

# The user may restrict the agent’s capabilities, but cannot expand them.
# Runtime knows:       read_file, write_file, delete_file, web_search
# Research agent gets: read_file, web_search
# User says:           don't access local files
# Effective tools:     web_search
# The user can remove read_file, but cannot add delete_file to the research agent.
# Similarly, A parent agent cannot give a subagent authority that the parent itself does not possess.

SYSTEM_PROMPT = """
You are a concise tool-using assistant.

Use an available tool whenever it can answer the user's request
more reliably than reasoning manually.

Never invent a tool result. Only report a tool result after the
runtime has returned an observation.

If no available tool can help, answer using your existing knowledge
when possible. Clearly state any limitation when the request requires
a capability or information you do not have.
""".strip()


def main():
    # Conversation state starts empty.
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    conversation_summary = ""
    summarized_until = 1  # summarization should start after the system prompt.

    print("Agent started. Type 'quit' to stop.")

    while True:
        user_message = input("\nYou: ").strip()

        if user_message.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        if not user_message:
            continue

        # Add the new user message to the existing conversation.
        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        conversation_summary, summarized_until = compact_old_messages(
            messages,
            conversation_summary,
            summarized_until,
            max_context_turns=MAX_CONTEXT_TURNS,
            compaction_batch_turns=COMPACTION_BATCH_TURNS,
        )

        final_answer, turn_usage = run_agent_turn(
            messages,
            conversation_summary,
            summarized_until,
            max_steps=MAX_STEPS,
        )

        print(f"\nAssistant: {final_answer}")
        print(f"[Messages stored in conversation: {len(messages)}]")
        
        print(
            "Turn usage: "
            f"model_calls={turn_usage['model_calls']}, "
            f"prompt_tokens={turn_usage['prompt_tokens']}, "
            f"completion_tokens={turn_usage['completion_tokens']}, "
            f"total_tokens={turn_usage['total_tokens']}, "
            f"model_seconds={turn_usage['model_seconds']:.2f}"
        )

if __name__ == "__main__":
    main()
