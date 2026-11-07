/*
 * Every stop reason this server can produce must be a value the OpenAI
 * chat-completions schema defines.
 *
 * It wasn't. The slot's internal vocabulary -- "deadline", "cancelled",
 * "shutdown" -- went straight onto the wire, and a strict client rejects the
 * whole response rather than the field. On 2026-09-27 the DeepSeek harness
 * reported:
 *
 *     This turn failed  Provider finish_reason: deadline  PI_AI_ERROR
 *
 * So a 46,346-token prompt that legitimately exhausted a 1,800 s deadline
 * surfaced to the operator as a protocol error, with no hint that the cause was
 * a prompt too long for the time allowed.
 *
 * The internal names are worth keeping -- /health counts deadline_stops
 * separately, and the logs need to say which limit was hit -- so the fix is a
 * mapping, and this is the test that stops a future reason escaping unmapped.
 */
#include "../mimo26_server_slot.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

/* https://platform.openai.com/docs/api-reference/chat/object -- finish_reason */
static const char *SCHEMA[] = {"stop", "length", "tool_calls",
                               "content_filter", "function_call"};

static int valid(const char *reason)
{
    for (size_t i = 0; i < sizeof SCHEMA / sizeof *SCHEMA; i++)
        if (!strcmp(reason, SCHEMA[i])) return 1;
    return 0;
}

int main(void)
{
    unsigned checks = 0;

    /*
     * Every enum value, not a hand-written list: adding a step without deciding
     * its wire form then fails here rather than reaching a client.
     */
    const mimo26_slot_step steps[] = {
        MIMO26_SLOT_CONTINUE, MIMO26_SLOT_STOP_CANCELLED,
        MIMO26_SLOT_STOP_DEADLINE, MIMO26_SLOT_STOP_LENGTH,
        MIMO26_SLOT_STOP_SHUTDOWN};
    for (size_t i = 0; i < sizeof steps / sizeof *steps; i++) {
        const char *internal = mimo26_slot_finish_reason(steps[i]);
        const char *wire = mimo26_slot_wire_finish_reason(internal);
        printf("  step %zu  internal %-10s -> wire %-10s %s\n", i, internal, wire,
               valid(wire) ? "ok" : "NOT IN SCHEMA");
        assert(valid(wire));
        ++checks;
    }

    /* The specific mappings, so a change of mind is deliberate. */
    assert(!strcmp(mimo26_slot_wire_finish_reason("deadline"), "length"));
    assert(!strcmp(mimo26_slot_wire_finish_reason("shutdown"), "length"));
    assert(!strcmp(mimo26_slot_wire_finish_reason("cancelled"), "stop"));
    checks += 3;

    /* Already-valid reasons pass through untouched. In particular "tool_calls"
     * is set as a literal on the tool path rather than derived from a step, so
     * a mapping that mangled it would break every tool call. */
    for (size_t i = 0; i < sizeof SCHEMA / sizeof *SCHEMA; i++) {
        assert(!strcmp(mimo26_slot_wire_finish_reason(SCHEMA[i]), SCHEMA[i]));
        ++checks;
    }

    /* Defensive, because the caller passes a variable that several paths
     * assign: a null must not crash the response builder. */
    assert(!strcmp(mimo26_slot_wire_finish_reason(NULL), "stop"));
    ++checks;

    /*
     * An UNRECOGNISED internal reason is passed through unchanged, which is a
     * deliberate choice: silently rewriting it to "stop" would claim a clean
     * finish for a stop nobody has classified. Passing it through keeps the
     * enum loop above as the thing that catches it.
     */
    assert(!strcmp(mimo26_slot_wire_finish_reason("some_future_reason"),
                   "some_future_reason"));
    assert(!valid(mimo26_slot_wire_finish_reason("some_future_reason")));
    checks += 2;

    printf("PASS %u finish-reason schema cases\n", checks);
    return 0;
}
