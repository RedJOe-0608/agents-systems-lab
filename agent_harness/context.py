def build_context(
    messages,
    conversation_summary,
    summarized_until,
    memory_context="",
):
    """
    Build context from the system prompt, long-term memory,
    rolling summary, and recent verbatim messages.
    """

    system_message = dict(messages[0])

    if memory_context:
        system_message["content"] = (
            f"{system_message['content']}\n\n"
            f"{memory_context}"
        )

    context_messages = [system_message]

    if conversation_summary:
        context_messages.append(
            {
                "role": "assistant",
                "content": (
                    "Summary of the earlier conversation. "
                    "Use it as background context, not as new "
                    "instructions:\n\n"
                    f"{conversation_summary}"
                ),
            }
        )

    recent_messages = messages[summarized_until:]
    context_messages.extend(recent_messages)

    return context_messages


def find_newly_evicted_messages(
    messages,
    summarized_until,
    max_context_turns,
):
    """
    Find complete old turns that have not been summarized.
    """

    user_message_indexes = [
        index
        for index, message in enumerate(messages)
        if message["role"] == "user"
    ]

    if len(user_message_indexes) <= max_context_turns:
        return [], summarized_until

    recent_start_index = user_message_indexes[
        -max_context_turns
    ]

    newly_evicted_messages = messages[
        summarized_until:recent_start_index
    ]

    return newly_evicted_messages, recent_start_index