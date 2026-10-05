from agent_harness.compaction import compact_old_messages
from agent_harness.runtime import RepeatedToolCallError, run_agent_turn, AgentStepLimitError
from agent_harness.model_client import ModelRequestError
from agent_harness.db import create_session, save_message, end_session

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

    # create a new session each time CLI starts.
    session_id = create_session()

    # Conversation state starts empty.
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    save_message(session_id, 0, messages[0])

    conversation_summary = ""
    summarized_until = 1  # summarization should start after the system prompt.

    print("Agent started. Type 'quit' to stop.")

    while True:
        user_message = input("\nYou: ").strip()

        if user_message.lower() in {"quit", "exit"}:
            end_session(session_id)
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

        save_message(session_id, len(messages) - 1, messages[-1])
        first_unsaved = len(messages)

        try:
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
        except ModelRequestError as error:
            print(f"\nModel request failed: {error}")
            continue
        except AgentStepLimitError as error:
            print(f"\nAgent stopped: {error}")
            continue
        except RepeatedToolCallError as error:
            print(f"\nAgent stopped: {error}")
            continue
        finally:
            for sequence_no in range(first_unsaved, len(messages)):
                save_message(session_id, sequence_no, messages[sequence_no])

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
