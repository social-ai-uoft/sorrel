# Development Workflow Summary Template

This document provides a template for documenting the development workflow of a feature integration or system enhancement. It captures the iterative refinement process, key decisions, and lessons learned.

## Initial Assessment Phase

**Typical Prompt Pattern**: "Is [Feature A] compatible with [System B]? Especially regarding [specific concern]"

**Purpose**: Identify potential compatibility issues or integration challenges before starting implementation.

**Key Activities**:
- Assess compatibility between new and existing systems
- Identify specific areas of concern
- Determine scope of integration work needed

## Initial Plan Request

**Typical Prompt Pattern**: "Make a plan for integrating [Feature A] with [System B]. [Specific requirements about inheritance, mapping, assignment logic]"

**Key Requirements to Capture**:
- What should be inherited from the original system?
- What mappings or data structures need to be created?
- What assignment or selection logic is needed?
- What randomization or filtering rules apply?

**Common Patterns**:
- Inheritance of configuration/mappings from training/source system
- Creation of lookup dictionaries or mappings
- Smart assignment based on available options
- Exclusion rules (e.g., exclude current/focal item)

## Plan Review Phase

**Typical Prompt Pattern**: "Review the plan to fix any issues, especially [specific concern like backward compatibility]"

**Focus Areas**:
- Backward compatibility with existing behavior
- Edge cases and error handling
- Performance considerations
- Code maintainability

## Clarification Requests

**Pattern 1: Understanding Default Behavior**
- **Prompt**: "For [entities/components] that don't have [attribute], what is the [encoding/representation]? Are they all [default value]?"
- **Follow-up**: "I mean [specific requirement for handling missing attributes]"
- **Outcome**: Clarify how to handle edge cases or missing data

**Pattern 2: Feature-Specific Details**
- **Prompt**: "In this revised plan, did you still keep [specific condition/feature]?"
- **Outcome**: Confirm preservation of existing functionality

## Major Refinement Request

**Typical Prompt Pattern**: "Check the current [system/environment] to see if anything needs to be changed in the plan. Ideally, we want [system] to do the following: (1) [condition A] (2) [condition B] (3) [condition C]. The [output format] should also include [additional fields] if not already included."

**Key Changes to Document**:
1. **Backward Compatibility**: Behavior when feature is disabled
2. **Feature Enabled**: New behavior when feature is active
3. **Output Format**: Changes to data export/reporting
4. **Iteration Logic**: How to loop through options/combinations

## Alignment Verification

**Typical Prompt Pattern**: "I want to double check to make sure we are aligned. When [feature] is enabled, [structure] is the same, with only [specific aspect] different. [Component] will not have [fixed attribute], but takes [dynamic attribute] from [source]. Is this consistent with the current plan?"

**Key Points to Verify**:
- Structure preservation (what stays the same)
- Scope of changes (what differs)
- Source of dynamic values (where they come from)
- What is NOT inherited (explicit exclusions)

## Compatibility Verification

**Typical Prompt Pattern**: "Double check to make sure [compatibility requirement] is ensured"

**Common Compatibility Checks**:
- Backward compatibility with existing code
- Method signature compatibility
- Data format compatibility
- API compatibility

## Attribute Source Clarification

**Typical Prompt Pattern**: "When [feature] is enabled, [attribute] still uses [source], not from [other source]"

**Purpose**: Clarify which attributes come from which sources, especially when multiple sources are available.

**Common Patterns**:
- Some attributes from training/source system
- Some attributes from test/current system setup
- Some attributes computed dynamically

## Visualization/Output Compatibility Check

**Typical Prompt Pattern**: "Check how [visualization/output] is assigned based on [criteria]. Check if the original setup is compatible with [feature] enabled, if not, how to change"

**Follow-up Pattern**: "I mean [specific requirement about what should remain the same]"

**Key Considerations**:
- What criteria determine the output (should remain same)
- What needs to change to prevent conflicts (e.g., filename uniqueness)
- What should be preserved for compatibility

## Final Code Review

**Typical Prompt Pattern**: "Review the current codespace and then see if the current plan has any errors"

**Common Issues to Look For**:
1. **Missing Flags/Configuration**: Plan checks for existence instead of using proper boolean flags
2. **Inconsistent Variable Usage**: Local variables instead of instance attributes
3. **Incomplete Implementations**: Placeholder comments instead of full code
4. **Incorrect Assumptions**: Assumptions about code structure that don't match reality

**Review Checklist**:
- [ ] All configuration flags are properly accessed
- [ ] Variable naming is consistent throughout
- [ ] All code examples are complete (no "same as above" placeholders)
- [ ] Assumptions match actual codebase structure
- [ ] Error handling is properly addressed

## Summary of Key Requirements Template

1. **Backward Compatibility**: Behavior when feature is disabled
   - What should remain unchanged?
   - What is the fallback behavior?

2. **Feature Enabled Behavior**: 
   - What should be inherited from source?
   - What iteration/looping logic is needed?
   - What dynamic values should be used?
   - What should NOT be inherited (explicit exclusions)?

3. **Output Format**: 
   - What additional columns/fields should be included?
   - What should be excluded from output?
   - What determines the values (source vs. computed)?

4. **Visualization/Display**:
   - What criteria determine the display (should remain same)?
   - What needs to change to prevent conflicts?
   - What should be preserved for compatibility?

5. **Edge Cases**: 
   - What special conditions need to be maintained?
   - How are missing/null values handled?

6. **Structure Preservation**: 
   - What remains the same?
   - What changes (scope of modifications)?

## Development Process Template

1. **Initial Assessment**: Identify compatibility concerns and integration scope
2. **Plan Creation**: Create initial integration plan with core requirements
3. **Iterative Refinement**: Multiple review cycles addressing:
   - Backward compatibility
   - Edge case handling
   - Iteration/looping logic
   - Output format
   - Visualization/display compatibility
   - Attribute source handling
4. **Code Review**: Final review against actual codebase to fix implementation errors

## Final State Checklist

The plan should correctly:
- [ ] Use configuration flags consistently throughout
- [ ] Maintain full backward compatibility
- [ ] Implement proper iteration/looping when feature enabled
- [ ] Include proper output format handling
- [ ] Preserve structure while updating specific aspects
- [ ] Handle all edge cases and missing values
- [ ] Match actual codebase structure and patterns

## Lessons Learned

**Common Pitfalls**:
1. Checking for existence instead of using boolean flags
2. Using local variables instead of instance attributes
3. Leaving incomplete implementations in plans
4. Making assumptions about code structure without verification

**Best Practices**:
1. Always verify against actual codebase before finalizing plan
2. Use consistent naming and access patterns
3. Complete all code examples (no placeholders)
4. Explicitly document what is and isn't inherited
5. Test backward compatibility explicitly
6. Clarify attribute sources early in the process

## Usage Notes

This template can be adapted for any feature integration or system enhancement workflow. Simply replace:
- Generic terms (Feature A, System B) with specific names
- Template prompts with actual prompts used
- Generic requirements with specific requirements
- Common patterns with actual patterns observed

The structure helps ensure:
- Comprehensive documentation of the development process
- Clear tracking of requirements and decisions
- Identification of common workflow patterns
- Lessons learned for future development
