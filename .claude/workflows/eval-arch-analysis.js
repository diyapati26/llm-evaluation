export const meta = {
  name: 'eval-arch-analysis',
  description: 'Deep-analyze both LLM-eval architectures (root evals/ + Simpler Arch), verify bug claims, and design a better paper-grade architecture',
  phases: [
    { title: 'Analyze', detail: 'parallel deep-readers: root arch, simpler arch, science/metrics, infra/reproducibility' },
    { title: 'Verify', detail: 'adversarially confirm each concrete bug claim against the real code' },
    { title: 'Design', detail: 'competing architecture proposals + judge panel' },
  ],
}

const BASE = 'C:/Users/SaiRudra/Desktop/llm-evaluation/Archive'

const ARCH_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    design_philosophy: { type: 'string', description: 'How this architecture is organized and its guiding principles' },
    file_map: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { path: {type:'string'}, role: {type:'string'} }, required: ['path','role'] } },
    strengths: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { title: {type:'string'}, detail: {type:'string'} }, required: ['title','detail'] } },
    weaknesses: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { title: {type:'string'}, detail: {type:'string'}, severity: {type:'string', enum:['high','med','low']} }, required: ['title','detail','severity'] } },
    bugs: { type: 'array', description: 'Concrete correctness bugs with file:line', items: { type: 'object', additionalProperties: false,
      properties: { title: {type:'string'}, file: {type:'string'}, location: {type:'string'}, detail: {type:'string'}, severity: {type:'string', enum:['high','med','low']}, confidence: {type:'string', enum:['high','med','low']} }, required: ['title','file','location','detail','severity','confidence'] } },
    keep: { type: 'array', items: {type:'string'}, description: 'Patterns/code worth carrying into a new architecture' },
    drop: { type: 'array', items: {type:'string'}, description: 'Patterns/code to discard' },
    extensibility_assessment: { type: 'string', description: 'How easy is it to add a model, provider, attack, or eval module?' },
  },
  required: ['design_philosophy','file_map','strengths','weaknesses','bugs','keep','drop','extensibility_assessment'],
}

const SCIENCE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    experimental_design_summary: { type: 'string' },
    metrics: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { name: {type:'string'}, definition: {type:'string'}, correctness: {type:'string'}, issues: {type:'string'} }, required: ['name','definition','correctness','issues'] } },
    threats_to_validity: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { title: {type:'string'}, detail: {type:'string'}, severity: {type:'string', enum:['high','med','low']} }, required: ['title','detail','severity'] } },
    statistical_issues: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { title: {type:'string'}, detail: {type:'string'}, severity: {type:'string', enum:['high','med','low']} }, required: ['title','detail','severity'] } },
    needed_for_publication: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { title: {type:'string'}, detail: {type:'string'}, priority: {type:'string', enum:['must','should','nice']} }, required: ['title','detail','priority'] } },
    dataset_assessment: { type: 'string' },
    bugs: { type: 'array', description: 'Metric/scoring correctness bugs with file:line', items: { type: 'object', additionalProperties: false,
      properties: { title: {type:'string'}, file: {type:'string'}, location: {type:'string'}, detail: {type:'string'}, severity: {type:'string', enum:['high','med','low']}, confidence: {type:'string', enum:['high','med','low']} }, required: ['title','file','location','detail','severity','confidence'] } },
  },
  required: ['experimental_design_summary','metrics','threats_to_validity','statistical_issues','needed_for_publication','dataset_assessment','bugs'],
}

const INFRA_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    caching: { type: 'string' },
    retry: { type: 'string' },
    cost_tracking: { type: 'string' },
    persistence: { type: 'string' },
    config: { type: 'string' },
    concurrency: { type: 'string' },
    reproducibility_gaps: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { title: {type:'string'}, detail: {type:'string'}, severity: {type:'string', enum:['high','med','low']} }, required: ['title','detail','severity'] } },
    bugs: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { title: {type:'string'}, file: {type:'string'}, location: {type:'string'}, detail: {type:'string'}, severity: {type:'string', enum:['high','med','low']}, confidence: {type:'string', enum:['high','med','low']} }, required: ['title','file','location','detail','severity','confidence'] } },
    recommendations: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { title: {type:'string'}, detail: {type:'string'} }, required: ['title','detail'] } },
  },
  required: ['caching','retry','cost_tracking','persistence','config','concurrency','reproducibility_gaps','bugs','recommendations'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    verdict: { type: 'string', enum: ['confirmed','refuted','partial'] },
    evidence: { type: 'string', description: 'Exact file:line and the code snippet that proves the verdict' },
    corrected_detail: { type: 'string', description: 'The accurate description after verification' },
    severity: { type: 'string', enum: ['high','med','low'] },
  },
  required: ['verdict','evidence','corrected_detail','severity'],
}

const DESIGN_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    approach_name: { type: 'string' },
    philosophy: { type: 'string' },
    module_layout: { type: 'string', description: 'Directory/file tree with one-line role per file' },
    data_model: { type: 'string', description: 'Core schemas/records and how they flow' },
    key_abstractions: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { name: {type:'string'}, purpose: {type:'string'}, why: {type:'string'} }, required: ['name','purpose','why'] } },
    extension_story: { type: 'string', description: 'Exact steps to add a new model, provider, attack type, and eval module' },
    reproducibility_story: { type: 'string' },
    tradeoffs: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { pro: {type:'string'}, con: {type:'string'} }, required: ['pro','con'] } },
    migration_path: { type: 'string', description: 'How to get from current code to this architecture' },
  },
  required: ['approach_name','philosophy','module_layout','data_model','key_abstractions','extension_story','reproducibility_story','tradeoffs','migration_path'],
}

const JUDGE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    ranking: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { approach_name: {type:'string'}, score: {type:'number'}, rationale: {type:'string'} }, required: ['approach_name','score','rationale'] } },
    winner: { type: 'string' },
    best_ideas_to_graft: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { idea: {type:'string'}, from_approach: {type:'string'} }, required: ['idea','from_approach'] } },
    synthesis_recommendation: { type: 'string', description: 'Concrete recommended architecture combining the winner with grafted ideas' },
  },
  required: ['ranking','winner','best_ideas_to_graft','synthesis_recommendation'],
}

// ───────────────────────── Phase 1: Analyze ─────────────────────────
phase('Analyze')

const rootPrompt = `You are a senior software architect auditing the ROOT (class-based) architecture of an LLM-evaluation research framework.

Read EVERY file under these paths and form a complete picture (paths have a space in "Simpler Arch" — quote them):
- ${BASE}/evals/ (all .py: cache.py, runner.py, schemas.py, datasets/loaders.py, datasets/manipulation_dataset.py, providers/base.py, providers/anthropic_provider.py, providers/openai_provider.py, providers/groq_provider.py, scorers/classical.py, scorers/llm_judge.py, modules/manipulation_runner.py)
- ${BASE}/run_eval.py
- ${BASE}/run_manipulation.py
- ${BASE}/configs/eval_config.yaml
- ${BASE}/tests/test_providers.py

This framework runs three eval modules (standard benchmark, manipulation resistance, moral/empathy) across multiple LLM providers. The research goal is a publishable paper on manipulation resistance.

Produce a rigorous structural and quality analysis. For bugs, give exact file + line/function and only report things you actually verified in the code (set confidence honestly). Assess: provider abstraction design, multi-turn conversation handling, caching, scoring, the manipulation runner, schema design, error handling, and how easy it is to extend. Identify what is worth keeping vs discarding for a clean rewrite. Output strictly as the required schema.`

const simplerPrompt = `You are a senior software architect auditing the "Simpler Arch" (function-based) architecture of an LLM-evaluation research framework.

Read EVERY file under "${BASE}/Simpler Arch/" (note the space — quote the path). Files include: cache.py, load_config.py, main.py, manipulation.py, moral.py, pricing.py, runner.py, run_manipulation.py, run_moral.py, schemas.py, scorers.py, providers/ (anthropic_provider.py, openai_provider.py, gemini_provider.py, groq_provider.py, openrouter_provider.py, conversation.py, retry.py, __init__.py), utils/ (load_dataset.py, load_local.py, stats.py), config/config.yaml, config/pricing.yaml, pricing.py.

This is the team's preferred consolidation target: function-based, no inheritance, Pydantic schemas, YAML-driven. It runs three eval modules (standard benchmark, manipulation, moral) across OpenAI/Anthropic/Groq/Gemini/OpenRouter. The research goal is a publishable paper on manipulation resistance.

Pay special attention to providers/conversation.py (the multi-turn Conversation hierarchy + chat() dispatcher), the provider asymmetry (OpenAI server-side state vs Anthropic/Groq/OpenRouter client-side), cache.py (note: the manipulation path reportedly does NOT use the cache — verify), retry.py, pricing, and scorers.py (resistance vector, multi-axis scoring). For bugs give exact file + line/function; only report verified findings (set confidence honestly). Identify keep vs discard for a clean rewrite. Output strictly as the required schema.`

const sciencePrompt = `You are a peer reviewer for a top-tier ML/AI-safety venue (think NeurIPS D&B / ACL) assessing the SCIENTIFIC METHODOLOGY of an LLM manipulation-resistance evaluation, BEFORE the paper is written. Be exacting.

Read these to understand the experiments, metrics, and current findings:
- ${BASE}/Simpler Arch/scorers.py and ${BASE}/evals/scorers/classical.py and ${BASE}/evals/scorers/llm_judge.py (scoring definitions in both architectures)
- ${BASE}/Simpler Arch/manipulation.py and ${BASE}/evals/modules/manipulation_runner.py (how attacks are run/scored)
- ${BASE}/Simpler Arch/moral.py (moral/empathy scoring)
- ${BASE}/Simpler Arch/utils/stats.py (statistics)
- ${BASE}/Simpler Arch/data/manipulation_attacks.jsonl, manipulation_drift.jsonl, moral_scenarios.jsonl (the stimuli)
- ${BASE}/Simpler Arch/RESULTS.md, ${BASE}/diya_report.md, ${BASE}/CHANGELOG.md (current findings + methodology notes)

Assess rigorously: (1) the experimental design (conditions: stateless/stateful/drift/neutral-control; the natural-drift correction; the resist/fold/hedge taxonomy; the "invalid = initially wrong, excluded" handling and the selection bias it creates); (2) each metric's definition and whether it is computed correctly; (3) statistical issues (sample sizes n=10, McNemar usage, multiple comparisons, phrasing-variant non-independence, LLM-judge as scorer reliability/bias); (4) threats to validity (using one provider as judge, attacker offering a fixed wrong answer, temperature 0 determinism vs single-sample, model-version drift); (5) what is REQUIRED vs nice-to-have to make this publishable (power/sample size, human validation of LLM judge, baselines, preregistration, inter-rater reliability, confidence intervals, effect sizes). For scoring/metric correctness bugs give exact file+line. Output strictly as the required schema.`

const infraPrompt = `You are an ML-infra/reproducibility engineer auditing an LLM-evaluation framework for paper-grade reproducibility and cost control. Two architectures coexist: root "${BASE}/evals/" (class-based) and "${BASE}/Simpler Arch/" (function-based).

Read and compare the infrastructure layers in BOTH:
- Caching: ${BASE}/evals/cache.py vs ${BASE}/Simpler Arch/cache.py (and how each runner uses or bypasses it — the manipulation path reportedly does NOT cache; verify in both run_manipulation.py files and manipulation.py / manipulation_runner.py)
- Retry/backoff: ${BASE}/Simpler Arch/providers/retry.py and any retry in root providers
- Cost tracking: ${BASE}/Simpler Arch/pricing.py + config/pricing.yaml + each provider's estimate_cost; root has hardcoded pricing — compare
- Persistence/checkpointing: incremental save logic in ${BASE}/Simpler Arch/run_manipulation.py and main.py; results/*.json + *_partial.jsonl
- Config: ${BASE}/Simpler Arch/load_config.py + config/config.yaml vs ${BASE}/configs/eval_config.yaml (note root config is empty)
- Concurrency: thread/async usage in both runners
- Determinism/reproducibility: temperature pinning, model-version logging, random seeds, dataset versioning

Assess each layer, list reproducibility gaps and infra bugs (exact file+line, honest confidence), and give concrete recommendations. The manipulation cache gap is known to be the biggest cost issue — quantify its impact (every run re-bills the API). Output strictly as the required schema.`

const [root, simpler, science, infra] = await parallel([
  () => agent(rootPrompt, { schema: ARCH_SCHEMA, phase: 'Analyze', label: 'root-arch' }),
  () => agent(simplerPrompt, { schema: ARCH_SCHEMA, phase: 'Analyze', label: 'simpler-arch' }),
  () => agent(sciencePrompt, { schema: SCIENCE_SCHEMA, phase: 'Analyze', label: 'methodology' }),
  () => agent(infraPrompt, { schema: INFRA_SCHEMA, phase: 'Analyze', label: 'reproducibility' }),
])

// ───────────────────────── Phase 2: Verify ─────────────────────────
phase('Verify')

const claims = []
const pushBugs = (obj, src) => {
  if (obj && Array.isArray(obj.bugs)) {
    for (const b of obj.bugs) {
      // verify only high/med severity OR low-confidence claims (the ones that matter or might be wrong)
      if (b.severity === 'high' || b.severity === 'med' || b.confidence !== 'high') {
        claims.push({ ...b, src })
      }
    }
  }
}
pushBugs(root, 'root-arch')
pushBugs(simpler, 'simpler-arch')
pushBugs(science, 'methodology')
pushBugs(infra, 'reproducibility')

log(`Verifying ${claims.length} concrete bug/issue claims against the real code`)

const CAP = 24
const toVerify = claims.slice(0, CAP)
if (claims.length > CAP) log(`Note: capping verification at ${CAP}; ${claims.length - CAP} lower-priority claims not independently re-verified`)

const verified = await parallel(toVerify.map((c) => () =>
  agent(`Adversarially verify this claimed bug/issue in the LLM-eval codebase. Default to skepticism — many plausible claims are wrong. Open the named file at the named location and READ the surrounding code before judging.

CLAIM (source: ${c.src}):
- Title: ${c.title}
- File: ${c.file}
- Location: ${c.location}
- Detail: ${c.detail}
- Claimed severity: ${c.severity}

Base path for files: ${BASE} (quote paths containing the space in "Simpler Arch"). Read the actual code, then decide: confirmed (the bug is real as described), refuted (it is not a bug / the description is wrong), or partial (real but mis-described or different severity). Provide the exact file:line and code snippet as evidence, a corrected accurate description, and the true severity. Output strictly as the required schema.`,
    { schema: VERDICT_SCHEMA, phase: 'Verify', label: `verify:${(c.file||'?').split('/').pop()}:${c.title.slice(0,28)}` })
    .then((v) => ({ claim: c, ...v }))
))

const confirmed = verified.filter(Boolean).filter((v) => v.verdict === 'confirmed' || v.verdict === 'partial')
log(`Verification complete: ${confirmed.length}/${verified.filter(Boolean).length} claims stand (confirmed or partial)`)

// ───────────────────────── Phase 3: Design ─────────────────────────
phase('Design')

const analysisContext = JSON.stringify({
  root_design: root?.design_philosophy,
  root_strengths: root?.strengths?.map(s => s.title),
  root_weaknesses: root?.weaknesses,
  root_keep: root?.keep, root_drop: root?.drop,
  simpler_design: simpler?.design_philosophy,
  simpler_strengths: simpler?.strengths?.map(s => s.title),
  simpler_weaknesses: simpler?.weaknesses,
  simpler_keep: simpler?.keep, simpler_drop: simpler?.drop,
  science_threats: science?.threats_to_validity,
  science_stat_issues: science?.statistical_issues,
  science_needed: science?.needed_for_publication,
  infra_gaps: infra?.reproducibility_gaps,
  infra_recos: infra?.recommendations,
  confirmed_bugs: confirmed.map(v => ({ title: v.claim.title, file: v.claim.file, severity: v.severity, detail: v.corrected_detail })),
}, null, 1)

const designAngles = [
  { name: 'Reproducibility-first', steer: 'Optimize above all for paper-grade reproducibility and auditability: content-addressed caching of EVERY call (including manipulation/multi-turn), a single append-only run-record (JSONL) as source of truth, deterministic seeds, full model-version + config provenance stamped on every record, and re-runnable analysis decoupled from data collection. The architecture should make "lost 2.5 hours of work to a crash" structurally impossible and make every number in the paper traceable to a cached raw response.' },
  { name: 'Extensibility-first', steer: 'Optimize for cleanly adding models, providers, attack types, eval modules, and scorers with zero core changes — registries/plugins, a uniform provider+conversation interface that hides server-side vs client-side state asymmetry, data-driven stimuli (JSONL), and config-driven everything. Favor the function-based Simpler Arch ethos but formalize the extension seams.' },
  { name: 'Scientific-rigor-first', steer: 'Optimize for the statistics and experimental design a reviewer demands: clean separation of (a) stimulus generation, (b) data collection, (c) scoring, (d) statistical analysis as four independent stages over a shared dataset; first-class handling of the resist/fold/hedge taxonomy, natural-drift correction, selection bias from "invalid" exclusions, multi-comparison correction, bootstrapped CIs, and LLM-judge validation against human labels. The code layout should mirror the analysis pipeline of the paper.' },
]

const proposals = await parallel(designAngles.map((a) => () =>
  agent(`You are designing the TARGET architecture for a rewrite of an LLM manipulation-resistance evaluation framework whose end goal is a publishable research paper. Two prior architectures exist: a class-based root "evals/" and a function-based "Simpler Arch/". Below is a structured analysis of both plus verified bugs and reviewer requirements.

DESIGN ANGLE — ${a.name}: ${a.steer}

You may also Read any file under ${BASE} to ground your proposal (quote paths with the space in "Simpler Arch"). Propose a concrete, buildable architecture (directory tree, data model, key abstractions, exact extension steps, reproducibility story, tradeoffs, and a migration path from the current code). Be specific and opinionated — this will be judged against competing proposals. Output strictly as the required schema.

ANALYSIS CONTEXT (JSON):
${analysisContext}`,
    { schema: DESIGN_SCHEMA, phase: 'Design', label: `design:${a.name}` })
))

const validProposals = proposals.filter(Boolean)

const judged = await agent(`You are the chief architect making the final call on the target architecture for an LLM manipulation-resistance evaluation framework intended to support a published research paper. Score each competing proposal (0-10) on: reproducibility/auditability, extensibility, scientific-pipeline fit, simplicity/maintainability, and migration cost. Pick a winner, list the best ideas to graft from the others, and write a concrete synthesis recommendation (the actual architecture to build: module tree, data model, key abstractions, extension story, reproducibility story). Be decisive and specific.

COMPETING PROPOSALS (JSON):
${JSON.stringify(validProposals, null, 1)}

KEY CONSTRAINTS FROM ANALYSIS (JSON):
${analysisContext}`,
  { schema: JUDGE_SCHEMA, phase: 'Design', label: 'judge-panel' })

return { root, simpler, science, infra, confirmed_bugs: confirmed, all_verdicts: verified.filter(Boolean), proposals: validProposals, judged }
