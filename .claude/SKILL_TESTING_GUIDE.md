# Research-Reproduction Skill Testing Guide

## Current Status
✅ Skill installed as **project-specific** in `.claude/skills/research-reproduction/`

This means the skill is ONLY available in this project folder, perfect for testing and iteration!

---

## How to Test the Skill

### 1. Restart Claude Code
The skill is installed but won't be loaded until you restart:

```bash
# Exit current session (Ctrl+C or Ctrl+D)
# Then restart in this project directory
cd /Users/aryateja/Desktop/Claude-WorkOnMac/Project-LeCoder/New-experiment-30dec
claude
```

### 2. Verify Skill is Loaded
After restarting, ask Claude:

```
What skills are available?
```

You should see `research-reproduction` in the list with its description.

### 3. Test the Skill
Try triggering it with various prompts:

**Test 1: Direct trigger**
```
I want to reproduce a research paper
```

**Test 2: Paper reproduction**
```
Help me implement the TITANS paper from arXiv
```

**Test 3: Algorithm extraction**
```
Extract the algorithm from this paper [attach PDF]
```

---

## How to Iterate and Improve

### Making Changes

1. **Edit the skill files** in `.claude/skills/research-reproduction/`
   - Main skill logic: `SKILL.md`
   - Templates: `templates/`
   - Scripts: `scripts/`
   - Prompts: `prompts/`

2. **Restart Claude Code** after each change
   ```bash
   # Exit and restart to reload the skill
   ```

3. **Test the changes** with the same prompts to verify improvements

### Common Improvements

| What to improve | File to edit | What to change |
|-----------------|-------------|----------------|
| Trigger phrases | `SKILL.md` line 3 | Update `description` field with better USE WHEN triggers |
| Workflow steps | `SKILL.md` sections | Modify phase instructions |
| Agent prompts | `prompts/*.md` | Refine extraction/implementation agents |
| Templates | `templates/*.md` | Improve output formats |
| Scripts | `scripts/*.py` | Add or fix automation |

### Debugging Tips

**Skill not triggering?**
- Check the `description` field has clear USE WHEN patterns
- Add more trigger keywords matching how users would ask
- Restart Claude Code to reload

**Skill triggers but doesn't work well?**
- Add detailed instructions to specific phases
- Refine agent prompts in `prompts/`
- Test with real paper PDFs to catch edge cases

**Need to see what's happening?**
- Add explicit outputs to workflow steps
- Use TodoWrite in the skill to track progress
- Check references in `references/` for patterns

---

## Promoting to Global (When Ready)

Once you've tested thoroughly and the skill works well:

### Option 1: Manual Copy (Recommended)
```bash
# Copy the entire tested skill to global
cp -r .claude/skills/research-reproduction ~/.claude/skills/

# Restart Claude Code
# The skill is now available in ALL projects!
```

### Option 2: Symbolic Link (Advanced)
```bash
# Link global to this location (changes here affect global)
ln -s "$(pwd)/.claude/skills/research-reproduction" ~/.claude/skills/research-reproduction
```

### After Promotion

1. **Keep project version** for continued testing if needed
2. **Test in other projects** to ensure it works globally
3. **Version control** - Consider adding to your PAI repo:
   ```bash
   cp -r ~/.claude/skills/research-reproduction ${PAI_DIR}/Skills/
   cd ${PAI_DIR}
   git add Skills/research-reproduction
   git commit -m "Add research-reproduction skill"
   ```

---

## Quick Reference

### File Structure
```
.claude/skills/research-reproduction/
├── SKILL.md                    # Main skill definition (edit triggers here)
├── README.md                   # Documentation
├── prompts/                    # Agent prompts
│   ├── extraction-agent.md
│   ├── implementation-agent.md
│   ├── verification-agent.md
│   └── documentation-agent.md
├── templates/                  # Output templates
│   ├── context-document.md
│   ├── implementation-plan.md
│   ├── readme-template.md
│   └── test-template.md
├── scripts/                    # UV automation scripts
│   ├── extract_paper.py
│   ├── quality_check.py
│   └── verify_equations.py
├── tools/                      # Tool documentation
│   ├── paper-intake.md
│   └── equation-extractor.md
└── references/                 # Reference patterns
    └── equation-patterns.md
```

### Testing Checklist

- [ ] Skill appears in "What skills are available?"
- [ ] Triggers correctly with "reproduce paper" prompts
- [ ] Extraction agents spawn properly
- [ ] Context documents generate correctly
- [ ] Implementation follows equation-first pattern
- [ ] Code quality checks work (ruff, ty, pytest)
- [ ] Documentation generates properly
- [ ] Git initialization works
- [ ] Works with real research papers
- [ ] No errors in multi-paper scenarios

---

## Next Steps

1. **Restart Claude Code now** to load the skill
2. **Test with a simple paper** to verify basic functionality
3. **Iterate on any issues** you find
4. **Promote to global** when satisfied
5. **Share back improvements** if you enhance the skill!

Happy testing! 🧪
