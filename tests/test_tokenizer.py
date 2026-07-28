from models.tokenizer import render_messages
def test_message_rendering():
 rendered=render_messages([{"role":"user","content":"hello"}],True)
 assert rendered=="<user>\nhello\n<assistant>\n"
