# NHSMM / SAE Governance and Release Model

## Core Guarantees

### nhsmm
- **Status**: Open source, forever.
- **Role**: Core algorithms, models, and research foundations.
- **Policy**:
  - Public development
  - Public issue tracking
  - No feature gating
- **Commitment**: Actively maintained and versioned independently of SAE.

---

## Research and Early Access Layer

### nhsmm-interfaces
- **Status**: Research and early-access layer.
- **Role**:
  - Interface experiments
  - Adapter sketches
  - Integration patterns
  - Design discussions and architectural reasoning
- **Lifecycle**:
  1. **Public Preview**  
     - Single showcase release
     - Demonstrates how SAE concepts integrate with `nhsmm`
     - Explicitly labeled as unstable and non-production
  2. **Closure After Preview**  
     - Repository archived
     - No further public commits
     - Remains readable as a historical reference

- **Access Model (Post-Preview)**:
  - Ongoing development moves behind Patreon
  - Access includes:
    - Research articles
    - Design notes
    - Interface sketches
    - Release snapshots of code
  - Focus is on **knowledge access**, not just source code

---

## Product Layer

### SAE (State Aware Engine)
- **Status**: Productized system.
- **Role**:
  - Stable interface contracts
  - Hardened adapters
  - Clear integration guarantees
- **Relationship to Research**:
  - Emerges from validated `nhsmm-interfaces` concepts
  - Incorporates lessons learned during research phase
- **Distribution**:
  - Separate branding and versioning
  - Clear compatibility matrix with `nhsmm`
  - May be commercial, licensed, or dual-licensed

---

## Separation of Concerns

| Layer | Purpose | Openness |
|------|---------|----------|
| nhsmm | Algorithms & models | Fully open |
| nhsmm-interfaces | Research & early access | Preview public → private |
| SAE | Stable product | Commercial / controlled |

---

## Trust and Transparency Principles
- No retroactive license changes.
- No removal of public previews.
- Clear messaging before any transition.
- Open research remains open in spirit via publications and explanations, even when code is gated.

---

## Outcome
- **Developers** get a stable, open foundation (`nhsmm`).
- **Early adopters** gain insight and influence via research access.
- **SAE** evolves into a coherent, professional product without destabilizing the open core.
