from typing import Dict


class MessageHistory:
    messages: [Dict[str, str]]
    user_role = "user"
    assistant_role = "assistant"

    def __init__(self):
        self.messages = []

    def __repr__(self):
        return str(self.messages)

    def add_user_message(self, message: str):
        self.messages.append({"role": self.user_role, "content": message})

    def append_to_last_user_message(self, message: str):
        if len(self.messages) == 0 or self.messages[-1]["role"] != self.user_role:
            # there is no history so far, or last message was not a user message
            self.add_user_message(message)
        else:
            self.messages[-1]["content"] += message

    def add_assistant_message(self, message: str):
        self.messages.append({"role": self.assistant_role, "content": message})

    def get_concatenated_messages(self):
        contents = [message["content"] for message in self.messages]

        return "\n".join(contents)
