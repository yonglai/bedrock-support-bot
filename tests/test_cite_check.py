import cite_check as c


def test_parse_sources_handles_several_blocks_in_one_string():
    got = c.parse_sources(
        ["[S1] (a.md) one\n[S2] (b.md) two\nmore two", "[S3] (c.md) three"]
    )
    assert got == {1: "(a.md) one", 2: "(b.md) two\nmore two", 3: "(c.md) three"}


def test_statements_split_and_attach_trailing_markers():
    got = c.statements(
        "Dental is 80% [S1]. Orthodontics is 50%. [S2]\n- Eye exam $100 [S3][S1]"
    )
    assert got[0] == ("Dental is 80% [S1].", [1])
    assert got[1][1] == [2]
    assert got[2][1] == [3, 1]


def test_uncited_statements_are_counted_not_judged(monkeypatch):
    class Boom:
        def converse(self, **kw):
            raise AssertionError("no cited statement, so no model call")

    monkeypatch.setattr(c, "rt", Boom())
    res = c.check("Hello there. Nothing cited here.", ["[S1] (a.md) x"])
    assert res == {"cited": 0, "uncited": 2, "unsupported": [], "missing": []}


def test_unsupported_statement_is_reported(monkeypatch):
    class Fake:
        def converse(self, **kw):
            body = kw["messages"][0]["content"][0]["text"]
            assert (
                "[S1] (a.md) 80%" in body and "[S2]" not in body
            )  # only the cited passage is shown
            return {
                "output": {
                    "message": {
                        "content": [
                            {
                                "text": '{"results": [{"i": 0, "supported": false, "reason": "says 80"}]}'
                            }
                        ]
                    }
                }
            }

    monkeypatch.setattr(c, "rt", Fake())
    res = c.check("Rate is 90% [S1].", ["[S1] (a.md) 80%\n[S2] (b.md) 90%"])
    assert (
        res["unsupported"] == [("Rate is 90% [S1].", "says 80")] and res["cited"] == 1
    )


def test_citation_to_a_missing_source_is_listed():
    res = c.check("Claim [S9].", ["[S1] (a.md) x"])
    assert res["missing"] == ["Claim [S9]."] and res["cited"] == 0
