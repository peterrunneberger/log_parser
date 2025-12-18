#! python3

from dataclasses import dataclass, field
from enum import StrEnum
import logging
import re
import sys

from argparse import ArgumentParser
from pathlib import Path
from typing import List

import yaml


class ParserError(Exception):
    """Exception for parser errors."""


class InvalidParserError(Exception):
    """Exception for invalid parser configurations."""


class PatternError(Exception):
    """Exception for pattern errors."""


def run_parser(log_files: List[Path], pattern_file: Path, output_file: Path, **kwargs):
    """Main function to run the log parser."""
    start_print = "# LogParser #"
    print("\n")
    print("#" * len(start_print))
    print(start_print)
    print("#" * len(start_print))
    print("\n")
    pattern = PatternLoader.load(pattern_file)
    log_parser = LogParser(log_files, pattern, output_file, **kwargs)
    log_parser.run()


class MatchResult(StrEnum):
    MATCH = "match"
    NO_MATCH = "no_match"
    INCOMPLETE = "incomplete_match"  # For sequence patterns
    ERROR = "error"


@dataclass
class PatternModel:
    group: str
    name: str
    pattern: List[str]
    pattern_type: str
    sequence: bool = False

    def __repr__(self):
        return f"{self.group}:{self.name}:{self.pattern_type}:Sequence={self.sequence}:{self.pattern}"


class PatternLoader:
    """Loader for pattern files."""

    @staticmethod
    def load(pattern_file: Path) -> List[PatternModel]:
        """Load patterns from a pattern file."""

        if not pattern_file.exists():
            raise FileNotFoundError(f"Pattern file {pattern_file} does not exist.")

        patterns = []

        with pattern_file.open("r", encoding="utf-8") as file:
            pattern_file = yaml.safe_load(file)

        for group in pattern_file:
            for name in pattern_file[group]:
                pattern_info = pattern_file[group][name]

                if not pattern_info.get("pattern_type", None):
                    raise PatternError(f"Pattern type missing for pattern {name} in group {group}.")
                if not pattern_info.get("pattern_type") == "regexp":
                    raise NotImplementedError("Only 'regexp' pattern type is implemented.")
                if not pattern_info.get("patterns", None):
                    raise PatternError(f"Patterns missing for pattern {name} in group {group}.")

                if pattern_info.get("sequence"):
                    pattern = PatternModel(
                        group=group,
                        name=name,
                        pattern=pattern_info.get("patterns"),
                        pattern_type=pattern_info.get("pattern_type"),
                        sequence=pattern_info.get("sequence", False)
                    )
                    patterns.append(pattern)
                else:
                    for pattern_str in pattern_info.get("patterns"):
                        pattern = PatternModel(
                            group=group,
                            name=name,
                            pattern=[pattern_str],
                            pattern_type=pattern_info.get("pattern_type")
                        )
                        patterns.append(pattern)

        print(f"Total patterns loaded: {len(patterns)}")
        for p in patterns:
            print(p)

        return patterns


@dataclass
class PatternMatch:
    """Class representing a pattern match in the log."""

    match_result: MatchResult
    file_name: str
    pattern: PatternModel
    content: dict = field(default_factory=dict)

    def __repr__(self):
        return "-" * 100 + "\n" + f"[{self.match_result}] {self.file_name}:{str(self.pattern)}\n{self.content}"


@dataclass
class LogParserResult:
    result_file: Path
    matches: List[PatternMatch] = field(default_factory=list)
    matches_dict: dict = None

    def create_matches_dict(self):
        """Create the matches dict."""
        pass

    def print_summary(self):
        """Print a summary of the parsing results."""
        # Dont forget partial matches for sequence patterns
        total_matches = len([m for m in self.matches if m.match_result == MatchResult.MATCH])
        total_no_matches = len([m for m in self.matches if m.match_result == MatchResult.NO_MATCH])
        total_incomplete = len([m for m in self.matches if m.match_result == MatchResult.INCOMPLETE])
        total_errors = len([m for m in self.matches if m.match_result == MatchResult.ERROR])
        summary_header = "# Result Summary #"
        print("\n")
        print("#" * len(summary_header))
        print(summary_header)
        print("#" * len(summary_header))
        print("\n")
        print(f"Total matches found: {total_matches}, Total no matches: {total_no_matches}, Total incomplete: {total_incomplete}, Total errors: {total_errors}")
        for match in self.matches:
            print(str(match))


class LogParser:
    """Log parser class."""

    def __init__(self, log_files: List[Path], pattern_list: List[PatternModel], output_file: Path, **kwargs):
        """Initialize the parser."""
        self.log_files = log_files
        self.pattern_list = pattern_list
        self.output_file = output_file
        self.result = LogParserResult(result_file=output_file)
        # TODO: Fix logging later
        # self.logger = logging.getLogger("LogParser")
        # log_level = logging.DEBUG if kwargs.get("verbose", False) else logging.INFO
        # self.logger.setLevel(log_level)
        # handler = logging.StreamHandler()
        # formatter = logging.Formatter('[%(levelname)s] %(message)s')
        # handler.setFormatter(formatter)
        # for handler in self.logger.handlers:
        #     self.logger.removeHandler(handler)
        # self.logger.addHandler(handler)

    def run(self):
        """Run the parser with the current filter on the log files."""

        def _find_sequence(file_name: str, lines: List[str], pattern: PatternModel) -> List[PatternMatch]:  # TODO: Fix logic in this function
            """Find sequence patterns in the log lines.

            while not at the end of the file
            step through all lines
            check if line matches first pattern
            if match, record result and step pattern
            continue to step through the lines with new pattern
            when done with all patterns, save that line number and look for more sequences of the patterns in the rest of the lines
            repeat this process
            """

            def _check_for_complete_sequence(sequence_results: list) -> MatchResult:
                if all(value == MatchResult.NO_MATCH for value in sequence_results):
                    return MatchResult.NO_MATCH
                if (any(value == MatchResult.MATCH for value in sequence_results) and
                    any(value != MatchResult.MATCH for value in sequence_results)):
                    return MatchResult.INCOMPLETE
                if all(value == MatchResult.MATCH for value in sequence_results):
                    return MatchResult.MATCH
                return MatchResult.ERROR

            matches = []
            no_content = ""
            no_line = "0"
            last_sequence_match_line = 0
            initial_lines_len = len(lines)
            done = False
            pattern_list = pattern.pattern
            while last_sequence_match_line < initial_lines_len or not done:
                result_current_sequence = []
                sequence_content = []
                pattern_no = 0
                for line_number, line in enumerate(lines, start=1):
                    try:
                        regex = re.compile(pattern_list[pattern_no])
                        if regex.match(line):
                            last_sequence_match_line = line_number
                            result_current_sequence.append(MatchResult.MATCH)
                            sequence_content.append({str(line_number): line.strip()})
                            pattern_no += 1
                            if pattern_no >= len(pattern_list):
                                last_sequence_match_line = line_number
                                break  # Found complete sequence, break to outer while loop
                    except Exception as e:
                        result_current_sequence.append(MatchResult.ERROR)
                        sequence_content.append({str(pattern_no): {no_line: str(e)}})
                match = PatternMatch(
                    match_result=_check_for_complete_sequence(result_current_sequence),
                    file_name=file_name,
                    pattern=pattern,
                    content=sequence_content
                )
                matches.append(match)
                if last_sequence_match_line == 0:
                    return matches  # Return early if no matches in the pattern was found the first time
                lines = lines[last_sequence_match_line:]
            return matches

        def _find_occurances(file_name: str, lines: List[str], pattern: PatternModel) -> List[PatternMatch]:
            """Find occurrence patterns in the log lines."""
            matches = []
            no_match = MatchResult.NO_MATCH
            no_content = ""
            try:
                regex = re.compile(pattern.pattern[0])
                for line_number, line in enumerate(lines, start=1):
                    if regex.match(line):
                        match = PatternMatch(
                            match_result=MatchResult.MATCH,
                            file_name=file_name,
                            pattern=pattern,
                            content={str(line_number): line.strip()}
                        )
                        matches.append(match)
            except Exception as e:
                no_match = MatchResult.ERROR
                no_content = str(e)
            if not matches:
                line_number = "0"
                matches = [
                    PatternMatch(
                        match_result=no_match,
                        file_name=file_name,
                        pattern=pattern,
                        content={line_number: no_content}
                    )
                ]
            return matches

        print(f"Starting log parsing of {len(self.log_files)} file(s).")
        for file in self.log_files:
            print(f"- {file}")

        for log in self.log_files:
            if not log.exists():
                raise FileNotFoundError(f"Log file {log} does not exist.")

        self.result.matches = []

        for file in self.log_files:
            print(f"Parsing log file: {file}")
            with file.open("r", encoding="utf-8") as log_file:
                lines = log_file.readlines()
                for pattern in self.pattern_list:
                    if pattern.sequence:
                        print(f"Searching for sequence pattern: {pattern.name}")
                        matches = _find_sequence(file, lines, pattern)
                    else:
                        print(f"Searching for all occurrences of pattern: {pattern.name}")
                        matches = _find_occurances(file, lines, pattern)
                    self.result.matches.extend(matches)

        self.result.print_summary()

    def __exit__(self, *_):
        """Exit handler to clean up resources."""
        print("Exiting log parser.")
        # self.result.create_matches_dict()
        # self.result.create_output_file()


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--log_files", "-f",
        type=Path,
        nargs="*",
        required=True,
        help="Path to the log file to be parsed."
    )
    arg_parser.add_argument(
        "--pattern_file", "-p",
        type=Path,
        required=True,
        help="Path to the pattern file defining parsing rules."
    )
    arg_parser.add_argument(
        "--output_file", "-o",
        type=Path,
        required=True,
        help="Path to the output file where parsed results will be saved."
    )
    arg_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging."
    )
    args = arg_parser.parse_args()
    sys.exit(run_parser(**vars(args)))
