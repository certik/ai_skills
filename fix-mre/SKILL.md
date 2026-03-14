---
name: fix-mre
description: >
  Fix an LFortran bug given an MRE (Minimal Reproducible Example) produced by
  the create-mre skill. Takes a path to a directory containing run.sh,
  reproduces the bug, fixes it in the LFortran source, adds an integration
  test, runs all tests, and commits the result. Triggers: fix MRE, fix bug, fix
  lfortran, fix reproducer, implement, resolve, patch, bugfix.
---

# Fix MRE — Fix an LFortran Bug from a Minimal Reproducible Example

Given an MRE (produced by the `create-mre` skill or manually), reproduce the
LFortran bug, fix it in the compiler source, add an integration test, verify
everything passes, and commit.

## Prerequisites

- The LFortran repository is the current working directory.
- The `lf` conda/pixi environment is available at `/Users/ondrej/.pixi/envs/lf/`.
- `lfortran` on PATH points to `src/bin/lfortran` (in-tree build).
- `flang` is available on PATH.
- `ninja` is the build tool (in-tree build in the repo root).

## Inputs to Gather

Ask the user for (if not already provided):

1. **Path to the MRE directory** — the directory containing `run.sh` and the
   `.f90` file(s) that reproduce the bug.

## Procedure

### Phase 0: Read Project Guidelines

Read the `AGENTS.md` file in the LFortran repository root to understand project
conventions, coding style, testing practices, and contribution guidelines. Also
read `CLAUDE.md` if it exists. Follow all instructions therein, but the
instructions in this SKILL.md file take precedence.

### Phase 1: Reproduce the Bug

1. Read `run.sh` in the MRE directory to understand the bug.
2. Run `run.sh` to confirm the failure:
   ```bash
   cd <mre-directory>
   bash run.sh
   ```
3. Note the **exact error message**, **error type** (compilation error, runtime
   crash, wrong output), and the **Fortran construct** involved.

### Phase 2: Diagnose the Root Cause

1. Analyze the error message to determine which compiler phase is failing:
   - **Parser** (`src/lfortran/parser/`): syntax errors, tokenizer failures
   - **Semantics** (`src/lfortran/semantics/`): type errors, symbol resolution
   - **ASR passes** (`src/libasr/pass/`): transformation errors
   - **Code generation** (`src/libasr/codegen/`): LLVM IR generation failures
2. Search the LFortran source for the error message text or error label to find
   the code that produces it.
3. Understand the code path that leads to the failure. Read surrounding code to
   understand the intended behavior.
4. Identify the minimal fix needed.

### Phase 3: Implement the Fix

1. Make the code change in the LFortran source. Keep changes minimal and
   focused on the bug.
3. Rebuild:
   ```bash
   eval "$(/Users/ondrej/miniforge3/bin/conda shell.bash activate /Users/ondrej/.pixi/envs/lf/)"
   ninja
   ```
4. Re-run the MRE to verify the fix:
   ```bash
   cd <mre-directory>
   bash run.sh
   ```
5. The MRE should now succeed (compile and/or run correctly) with `lfortran`.

If the fix doesn't work, iterate: re-diagnose, adjust, rebuild, and re-test.

### Phase 4: Add an Integration Test

1. Look at existing integration tests in `integration_tests/` to find similar
   tests (same Fortran construct, similar naming pattern).
2. Create a new `.f90` file in `integration_tests/` following the naming
   convention (e.g., `intrinsic_name_NN.f90`, `derived_type_feature_NN.f90`).
   Pick the next available number.
3. The test should be based on the MRE but written as a proper integration test:
   - Include runtime checks using `if (result /= expected) error stop` idioms.
   - Print results for debugging: `print *, value`.
   - Keep it minimal but cover the bug scenario.
4. Register the test in `integration_tests/CMakeLists.txt`:
   - Find the appropriate section (alphabetical or grouped by feature).
   - Add a `RUN(NAME <test_name> LABELS gfortran llvm)` style entry.
   - Use at least the labels `gfortran` and `llvm`.
5. Verify the test compiles with `flang`:
   ```bash
   flang -o /tmp/test_flang integration_tests/<test_name>.f90 && /tmp/test_flang
   rm -f /tmp/test_flang
   ```
6. Verify the test compiles and runs with `lfortran`:
   ```bash
   cd integration_tests
   eval "$(/Users/ondrej/miniforge3/bin/conda shell.bash activate /Users/ondrej/.pixi/envs/lf/)"
   ./run_tests.py -t <test_name>
   ```

### Phase 5: Run Integration Tests

Run the full integration test suite:

```bash
eval "$(/Users/ondrej/miniforge3/bin/conda shell.bash activate /Users/ondrej/.pixi/envs/lf/)"
cd integration_tests
./run_tests.py -j16 &> log
tail -n30 log
```

- If all tests pass, proceed to Phase 6.
- If any test fails:
  1. Examine the log to identify which test failed and why.
  2. Determine if the failure is caused by your change (a regression) or a
     pre-existing issue.
  3. If your change caused it, fix the regression, rebuild (`ninja` in root),
     and re-run tests.
  4. Repeat until all integration tests pass.

### Phase 6: Run Reference Tests

Return to the LFortran root and run reference tests:

```bash
eval "$(/Users/ondrej/miniforge3/bin/conda shell.bash activate /Users/ondrej/.pixi/envs/lf/)"
cd <lfortran-root>
./run_tests.py
```

- If reference tests pass, proceed to Phase 7.
- If reference tests fail (expected when your fix changes compiler output):
  1. Update reference results:
     ```bash
     ./run_tests.py -u
     ```
  2. Review the changes with `git diff` to ensure all reference updates are
     correct and expected — they should all be consequences of your bug fix,
     not regressions.
  3. If any reference change looks wrong, investigate and fix before proceeding.

### Phase 7: Commit

1. Stage all new and changed files:
   ```bash
   git add -A
   ```
2. Review what will be committed:
   ```bash
   git diff --cached --stat
   ```
3. Write a commit message:
   - First line: imperative mood, ≤50 characters, describing the fix
     (e.g., `Fix array reshape with allocatable components`)
   - Blank line, then a body explaining:
     - What the bug was
     - What the fix does
     - That an integration test was added
   - Do NOT include `Co-authored-by` trailers.
4. Commit:
   ```bash
   git commit
   ```

### Phase 8: Summary

Print a summary for the user:

```
Bug fixed and committed!

Fix: <one-line description of what was changed>
Files modified:
  <list of changed source files>
Integration test: integration_tests/<test_name>.f90
Commit: <short SHA and first line of commit message>

All integration tests and reference tests pass.
```

## Tips

- **Multiple errors**: The MRE may expose more than one bug. First fix the bug
  that the MRE demonstrates, commit. If you discover additional issues to fully
  compile and run the newly added test, fix them as well.
- **ASR passes**: Many bugs live in ASR passes (`src/libasr/pass/`). These
  transform the ASR tree and are a common source of codegen failures.
- **Semantic errors**: If the bug is "not yet implemented", the fix likely
  involves adding a new case in the semantics or codegen visitor.
- **Error messages**: Search for the error text with `grep -r "error text"
  src/` to quickly find where the error is raised.
- **Modfile issues**: If you see "Incompatible format: LFortran Modfile...",
  run `ninja clean && ninja` to rebuild from scratch.
- **Rebuild quickly**: `ninja` only recompiles changed files. Use it
  frequently during iteration.
- **Test naming**: Look at nearby tests in `CMakeLists.txt` for naming
  conventions. Usually it's `<feature>_<number>`.
