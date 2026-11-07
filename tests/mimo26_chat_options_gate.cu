// Request-option parsing for /v1/chat/completions, CPU-only: the actual
// parse_chat from the server, no GPU, no sockets, no worker.
//
// This gate exists because chat_template_kwargs was silently dropped. The
// capability behind it worked -- top-level enable_thinking rendered the empty
// think block correctly -- so nothing failed loudly, and a caller asking for
// no thinking got thinking with a 200. Every case below is about the
// difference between refusing an option and ignoring it.
#include <initializer_list>
#define main mimo26_unused_server_main
#include "../mimo26_server.cu"
#undef main
#include <cassert>
#include <cstring>
#include <string>

namespace {

struct Outcome {
    bool accepted;
    bool thinking;
    std::string code;
    std::string error;
};

Outcome run(const std::string &body)
{
    chat_request request{};
    char code[64] = {0};
    char error[256] = {0};
    const bool ok = parse_chat(body.data(), body.size(), &request,
                               code, sizeof code, error, sizeof error);
    Outcome out{ok, request.enable_thinking, code, error};
    if (ok) {
        chat_request_free(&request);
    }
    return out;
}

const char *MSG = R"("messages":[{"role":"user","content":"hi"}])";

std::string body(const std::string &extra)
{
    return std::string("{") + MSG + (extra.empty() ? "" : "," + extra) + "}";
}

void accepts(const std::string &extra, bool expect_thinking)
{
    const Outcome o = run(body(extra));
    if (!o.accepted) {
        fprintf(stderr, "expected accept for %s: %s (%s)\n",
                extra.c_str(), o.error.c_str(), o.code.c_str());
        assert(false);
    }
    if (o.thinking != expect_thinking) {
        fprintf(stderr, "expected enable_thinking=%d for %s, got %d\n",
                (int)expect_thinking, extra.c_str(), (int)o.thinking);
        assert(false);
    }
}

void refuses(const std::string &extra, const char *expect_code)
{
    const Outcome o = run(body(extra));
    if (o.accepted) {
        fprintf(stderr, "expected refusal for %s, was accepted\n",
                extra.c_str());
        assert(false);
    }
    if (o.code != expect_code) {
        fprintf(stderr, "expected code %s for %s, got %s (%s)\n",
                expect_code, extra.c_str(), o.code.c_str(), o.error.c_str());
        assert(false);
    }
    assert(!o.error.empty());
}

} // namespace

int main()
{
    // Default: thinking on, with neither spelling present.
    accepts("", true);

    // Top-level spelling, which already worked and must keep working.
    accepts(R"("enable_thinking":false)", false);
    accepts(R"("enable_thinking":true)", true);

    // The defect: the nested spelling every OpenAI-compatible client sends.
    accepts(R"("chat_template_kwargs":{"enable_thinking":false})", false);
    accepts(R"("chat_template_kwargs":{"enable_thinking":true})", true);
    accepts(R"("chat_template_kwargs":{})", true);

    // Both spellings, agreeing, is fine; disagreeing is refused rather than
    // resolved, because either resolution discards a stated intent.
    accepts(R"("enable_thinking":false,"chat_template_kwargs":{"enable_thinking":false})",
            false);
    accepts(R"("enable_thinking":true,"chat_template_kwargs":{"enable_thinking":true})",
            true);
    refuses(R"("enable_thinking":true,"chat_template_kwargs":{"enable_thinking":false})",
            "invalid_request");
    refuses(R"("enable_thinking":false,"chat_template_kwargs":{"enable_thinking":true})",
            "invalid_request");

    // A misspelled or unknown kwarg must fail loudly. Silently accepting
    // "enable_think" is exactly the bug this gate is here for: the caller
    // would get thinking and a 200.
    refuses(R"("chat_template_kwargs":{"enable_think":false})",
            "option_unsupported");
    refuses(R"("chat_template_kwargs":{"add_generation_prompt":false})",
            "option_unsupported");
    refuses(R"("chat_template_kwargs":{"enable_thinking":false,"tools":[]})",
            "option_unsupported");

    // The named key comes back to the caller, so a typo is diagnosable.
    {
        const Outcome o = run(body(
            R"("chat_template_kwargs":{"enable_thinkingg":false})"));
        assert(!o.accepted);
        assert(o.error.find("enable_thinkingg") != std::string::npos);
    }

    // Type errors are refused, not coerced. "false" is not false.
    refuses(R"("chat_template_kwargs":"enable_thinking")", "invalid_request");
    refuses(R"("chat_template_kwargs":[])", "invalid_request");
    refuses(R"("chat_template_kwargs":{"enable_thinking":"false"})",
            "invalid_request");
    refuses(R"("chat_template_kwargs":{"enable_thinking":0})",
            "invalid_request");
    refuses(R"("chat_template_kwargs":{"enable_thinking":null})",
            "invalid_request");
    refuses(R"("enable_thinking":"false")", "invalid_request");
    refuses(R"("enable_thinking":1)", "invalid_request");

    // An over-long key must be truncated into the message, not overrun it.
    {
        const std::string long_key(400, 'k');
        const Outcome o = run(body(R"("chat_template_kwargs":{")" + long_key +
                                   R"(":false})"));
        assert(!o.accepted);
        assert(o.code == "option_unsupported");
        assert(o.error.size() < 256);
    }

    // Unchanged: sampling options stay refused, and unknown top-level fields
    // stay ignored. Rejecting every unknown top-level key would break clients
    // that legitimately send "user" or "stream_options" for no safety gain,
    // so that is deliberately not what this fix does.
    refuses(R"("temperature":0)", "option_unsupported");
    refuses(R"("top_p":1)", "option_unsupported");
    accepts(R"("user":"alex","stream_options":{"include_usage":true})", true);

    printf("mimo26_chat_options_gate: all cases pass\n");
    return 0;
}
