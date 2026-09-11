"""Unit tests for delta-cards CLI subcommands and entrypoints."""

import pytest
import sys
from delta_cards.cli import parse_args, build_parser, main_scrape, main_assign


def test_cli_default_is_all_subcommand():
    args = parse_args([])
    assert args.subcommand == "all"
    assert args.skip_tcgdex is False
    assert args.scrape_prices is False
    assert args.profile is None
    assert args.type is None


def test_cli_scrape_subcommand_parsing():
    args = parse_args(["scrape", "-p", "single_card", "-l", "10", "--loop", "--interval", "120"])
    assert args.subcommand == "scrape"
    assert args.profile == ["single_card"]
    assert args.limit == 10
    assert args.loop is True
    assert args.interval == 120
    assert args.no_db is False


def test_cli_assign_subcommand_parsing():
    args = parse_args(["assign", "-l", "25", "--scrape-prices", "--loop", "--interval", "45"])
    assert args.subcommand == "assign"
    assert args.limit == 25
    assert args.scrape_prices is True
    assert args.loop is True
    assert args.interval == 45
    assert args.once is False

    once_args = parse_args(["assign", "--once"])
    assert once_args.subcommand == "assign"
    assert once_args.once is True


def test_cli_all_subcommand_parsing():
    args = parse_args(["all", "-p", "booster_box", "--scrape-prices"])
    assert args.subcommand == "all"
    assert args.profile == ["booster_box"]
    assert args.scrape_prices is True


def test_cli_root_backward_compatibility_flags():
    args = parse_args(["--skip-tcgdex", "--assign-existing-cards", "-p", "single_card"])
    assert args.subcommand == "all"
    assert args.skip_tcgdex is True
    assert args.assign_existing_cards is True
    assert args.profile == ["single_card"]


def test_main_scrape_sys_argv_injection(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["delta-cards-scrape", "-p", "single_card"])
    captured_args = []
    
    def fake_main():
        from delta_cards.cli import parse_args
        captured_args.append(parse_args())

    monkeypatch.setattr("delta_cards.cli.main", fake_main)
    main_scrape()
    assert sys.argv[1] == "scrape"


def test_main_assign_sys_argv_injection(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["delta-cards-assign", "-l", "5"])
    captured_args = []

    def fake_main():
        from delta_cards.cli import parse_args
        captured_args.append(parse_args())

    monkeypatch.setattr("delta_cards.cli.main", fake_main)
    main_assign()
    assert sys.argv[1] == "assign"
