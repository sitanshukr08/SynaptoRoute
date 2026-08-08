from synaptoroute.cli import command_info, command_match

def test_cli_command_info(capsys):
    command_info()
    captured = capsys.readouterr()
    assert "SynaptoRoute System Information" in captured.out
    assert "SynaptoRoute Version" in captured.out

def test_cli_command_match(capsys, monkeypatch, fake_encoder):
    monkeypatch.setattr(
        "synaptoroute.encoder.FastEmbedEncoder",
        lambda threads=None: fake_encoder,
    )
    command_match("billing issue")
    captured = capsys.readouterr()
    assert "Evaluating query:" in captured.out
    assert "Matched Route  : billing" in captured.out
    assert "Decision Reason: matched" in captured.out
