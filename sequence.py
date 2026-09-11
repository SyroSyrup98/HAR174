# Prototype experiment sequence
# This is configurable and is NOT the official ISRO experiment sequence.

EXPERIMENT_SEQUENCE = [
    "bottle",
    "book",
    "cup"
]


class SequenceValidator:

    def __init__(self, sequence):
        self.sequence = sequence
        self.current_step = 0

    def expected_object(self):
        if self.current_step >= len(self.sequence):
            return None

        return self.sequence[self.current_step]

    def process_event(self, detected_object):

        expected = self.expected_object()

        if expected is None:
            return "COMPLETE"

        if detected_object == expected:

            self.current_step += 1

            if self.current_step >= len(self.sequence):
                return "COMPLETE"

            return "PASS"

        else:
            return "DEVIATION"


# Simple test
if __name__ == "__main__":

    validator = SequenceValidator(EXPERIMENT_SEQUENCE)

    print("Expected:", validator.expected_object())

    print("Detected bottle:", validator.process_event("bottle"))
    print("Expected:", validator.expected_object())

    print("Detected cup:", validator.process_event("cup"))
    print("Expected:", validator.expected_object())

    print("Detected book:", validator.process_event("book"))
    print("Expected:", validator.expected_object())

    print("Detected cup:", validator.process_event("cup"))
    print("Expected:", validator.expected_object())