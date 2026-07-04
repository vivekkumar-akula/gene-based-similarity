#!/usr/bin/env python3
# =============================================================================
# Disease Similarity Scoring on Gene and Disease Networks Using node2vec
# Project-1 : Human Genes and Gene Relationships with Diseases
#
# SINGLE-FILE program with three modes (subcommands):
#   run    -> compute similarity(D1, D2) in [0,1]         (the main pipeline)
#   fetch  -> collect real protein sequences from NCBI    (link 1, for BLAST)
#   viz    -> generate the result figures (PNG)
#
# Uses all THREE CTD downloads:
#   #7  CTD_curated_genes_diseases.csv  -> disease<->gene backbone   (required)
#   #18 CTD_genes.csv                   -> UniProt for BLAST (A6)     (optional)
#   #16 CTD_diseases.csv                -> MeSH hierarchy validation (C4)(optional)
#
# Methodology (classes / subclasses):
#   A. DATA PROCESSING   A1 collect inputs  A2 parse & integrate  A3 resolve D1,D2
#                        A4 gene significance weight
#                        A5 sequence/BLAST similarity (real blastp or k-mer fallback)
#                        A6 gene ID mapping (NCBI . OMIM . UniProt)
#   B. DATA MODELLING    B1 build graph  B2 biased random walks  B3 skip-gram
#   C. DATA EVALUATION   C1 baselines  C2 similarity scoring  C3 calibration & banding
#                        C4 validation vs MeSH disease hierarchy (#16)
#
# Examples:
#   python3 project1_disease_similarity.py run
#   python3 project1_disease_similarity.py run "Breast Neoplasms" "Pancreatic Neoplasms"
#   python3 project1_disease_similarity.py run "Breast Neoplasms" "Pancreatic Neoplasms" --blast
#   python3 project1_disease_similarity.py fetch "Breast Neoplasms" "Pancreatic Neoplasms"
#   python3 project1_disease_similarity.py viz --fast
# =============================================================================
import sys, os, csv, io, math, argparse, random, time
from collections import defaultdict
import numpy as np
import networkx as nx
from gensim.models import Word2Vec

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_CTD      = os.path.join(HERE, "inputs", "CTD_curated_genes_diseases.csv")
DEF_OMIM     = os.path.join(HERE, "inputs", "omim_genemap.tsv")
DEF_FASTA    = os.path.join(HERE, "inputs", "ncbi_gene_proteins.fasta")
DEF_GENES    = os.path.join(HERE, "inputs", "CTD_genes.csv")           # #18 A6 UniProt
DEF_OMIMGENE = os.path.join(HERE, "inputs", "omim_gene_numbers.tsv")   # A6 OMIM gene numbers
DEF_DISVOCAB = os.path.join(HERE, "inputs", "CTD_diseases.csv")        # #16 C4 validation
FIGDIR       = os.path.join(HERE, "figures")

# #############################################################################
# A.  DATA PROCESSING
# #############################################################################

# ---- A4  Gene significance weight -------------------------------------------
def gene_significance_weight(pubmed_ids, evidence):
    n_pmid = len([p for p in pubmed_ids.split("|") if p.strip()])
    w = 1.0 + math.log1p(n_pmid)
    if "marker/mechanism" in evidence:
        w += 0.5
    return w

# ---- A1+A2  Collect CTD input & parse/integrate into a disease->gene map -----
# Fields: GeneSymbol,GeneID,DiseaseName,DiseaseID,DirectEvidence,OmimIDs,PubMedIDs
def load_ctd(path):
    dis_genes = defaultdict(dict)
    name2id   = defaultdict(set)
    id2name   = {}
    omim2mesh = defaultdict(set)
    gene_deg  = defaultdict(int)
    gene2id   = {}
    rows = [l for l in open(path, encoding="utf-8") if not l.startswith("#")]
    for r in csv.reader(io.StringIO("".join(rows))):
        if len(r) < 7: continue
        gs, gid, dn, did, ev, omim, pmids = r[:7]
        w = gene_significance_weight(pmids, ev)                 # A4
        dis_genes[did][gs] = max(dis_genes[did].get(gs, 0.0), w)
        gene2id.setdefault(gs, gid)                            # A6 NCBI id
        id2name[did] = dn
        name2id[dn.lower()].add(did)
        for o in omim.split("|"):
            if o.strip(): omim2mesh[o.strip()].add(did)
    for did, genes in dis_genes.items():
        for g in genes: gene_deg[g] += 1
    return dict(dis_genes), name2id, id2name, omim2mesh, gene_deg, gene2id

# ---- A1+A2  Collect OMIM input (link 2) -> extra disease->gene edges ---------
def load_omim(path):
    omim_genes = defaultdict(dict)
    omim_name  = {}
    if not os.path.exists(path): return {}, {}
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or line.startswith("PhenotypeMIM"): continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 4: continue
        mim, pname, gs, gid = (x.strip() for x in p[:4])
        node = "OMIM:" + mim
        omim_genes[node][gs] = 1.5
        omim_name[node] = pname
    return dict(omim_genes), omim_name

# ---- A1  Collect NCBI input (link 1) -> gene protein sequences ---------------
def load_fasta(path):
    seqs, sym, buf = {}, None, []
    if not os.path.exists(path): return seqs
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith(";"): continue
        if line.startswith(">"):
            if sym: seqs[sym] = "".join(buf)
            sym = line[1:].split("|")[0].strip()
            buf = []
        else:
            buf.append(line.strip())
    if sym: seqs[sym] = "".join(buf)
    return seqs

# ---- A1+A2  Disease vocabulary (#16 MEDIC) -> MeSH categories, for C4 --------
# Fields: DiseaseName,DiseaseID,AltDiseaseIDs,Definition,ParentIDs,TreeNumbers,...
def load_disease_vocab(path):
    """DiseaseID -> set of top-level MeSH categories (e.g. {'C04','C17'})."""
    did2tree = {}
    if not os.path.exists(path): return did2tree
    csv.field_size_limit(10_000_000)
    with open(path, encoding="utf-8") as fh:
        for r in csv.reader(l for l in fh if not l.startswith("#")):
            if len(r) < 6 or r[1] == "DiseaseID": continue
            cats = set()
            for t in r[5].split("|"):
                t = t.strip()
                if t: cats.add(t.split("/")[0].split(".")[0])   # top category, e.g. C04
            if cats: did2tree[r[1]] = cats
    return did2tree

# ---- A5  Sequence / BLAST similarity -----------------------------------------
# (a) REAL BLAST -> blast_homology_edges() : needs NCBI BLAST+ on PATH.
# (b) k-mer cosine -> seq_similarity()      : offline fallback.
def kmer_vec(seq, k=3):
    v = defaultdict(float)
    for i in range(len(seq) - k + 1):
        v[seq[i:i+k]] += 1.0
    return v
def seq_similarity(a, b, k=3):
    va, vb = kmer_vec(a, k), kmer_vec(b, k)
    if not va or not vb: return 0.0
    dot = sum(va[x]*vb.get(x, 0.0) for x in va)
    na  = math.sqrt(sum(x*x for x in va.values()))
    nb  = math.sqrt(sum(x*x for x in vb.values()))
    return dot/(na*nb) if na and nb else 0.0

def blast_available():
    import shutil
    return bool(shutil.which("makeblastdb") and shutil.which("blastp"))

def blast_homology_edges(seqs, thr=0.30, evalue=10.0):
    """REAL all-vs-all blastp -> [(g1,g2,sim)], sim = normalized bit-score in [0,1].
    Returns None if BLAST+ is not installed (caller falls back to k-mer)."""
    import subprocess, tempfile
    if not blast_available():
        return None
    with tempfile.TemporaryDirectory() as td:
        fa = os.path.join(td, "genes.faa"); db = os.path.join(td, "db")
        with open(fa, "w") as f:
            for g, s in seqs.items():
                if s: f.write(f">{g}\n{s}\n")
        subprocess.run(["makeblastdb", "-in", fa, "-dbtype", "prot", "-out", db],
                       check=True, capture_output=True)
        res = subprocess.run(
            ["blastp", "-query", fa, "-db", db, "-evalue", str(evalue),
             "-outfmt", "6 qseqid sseqid pident bitscore"],
            check=True, capture_output=True, text=True).stdout
    self_bit, pairs = {}, {}
    for line in res.splitlines():
        q, s, pid, bit = line.split("\t"); bit = float(bit)
        if q == s: self_bit[q] = max(self_bit.get(q, 0.0), bit)
    for line in res.splitlines():
        q, s, pid, bit = line.split("\t")
        if q >= s: continue
        norm = float(bit) / max(self_bit.get(q, 1.0), self_bit.get(s, 1.0), 1.0)
        pairs[(q, s)] = max(pairs.get((q, s), 0.0), min(norm, 1.0))
    return [(a, b, v) for (a, b), v in pairs.items() if v >= thr]

# ---- A3  Resolve a user query (name / MeSH id / OMIM number) -> graph node ---
def resolve(query, name2id, id2name, omim2mesh, omim_name, dis_genes, G):
    q = query.strip()
    if q.startswith("OMIM:") or q.isdigit():
        mim = q.split(":")[-1]
        if mim in omim2mesh:
            best = max(omim2mesh[mim], key=lambda d: len(dis_genes.get(d, {})))
            return best, id2name.get(best, best)
        if ("OMIM:"+mim) in G:
            return "OMIM:"+mim, omim_name.get("OMIM:"+mim, "OMIM:"+mim)
    if q in G: return q, id2name.get(q, q)
    ql = q.lower()
    if ql in name2id:
        best = max(name2id[ql], key=lambda d: len(dis_genes.get(d, {})))
        return best, id2name[best]
    cand = [(d, id2name[d], len(dis_genes.get(d, {})))
            for nm, ids in name2id.items() if ql in nm for d in ids]
    if cand:
        best = max(cand, key=lambda x: x[2]); return best[0], best[1]
    return None, None

def resolve_name_only(query, dis_genes, name2id, id2name):
    ql = query.strip().lower()
    cand = [d for nm, ids in name2id.items() if ql in nm for d in ids]
    return max(cand, key=lambda d: len(dis_genes.get(d, {}))) if cand else None

# ---- A6  Gene ID mapping: NCBI Entrez . OMIM gene # . UniProt (for BLAST) ----
def load_gene_vocab(path):
    vocab = {}
    if not os.path.exists(path): return vocab
    csv.field_size_limit(10_000_000)
    with open(path, encoding="utf-8") as fh:
        for r in csv.reader(fh):
            if not r or r[0].startswith("#") or len(r) < 8: continue
            sym, name, gid, uni = r[0], r[1], r[2], r[7].split("|")[0]
            vocab[sym] = {"geneid": gid, "uniprot": uni, "name": name}
    return vocab

def load_omim_gene_numbers(path):
    mp = {}
    if not os.path.exists(path): return mp
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or line.startswith("GeneSymbol"): continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[0].strip(): mp[p[0].strip()] = p[1].strip()
    return mp

def gene_id_table(genes, gene2id, gene_vocab, omim_gene):
    rows = []
    for g in sorted(genes):
        ncbi = gene2id.get(g) or gene_vocab.get(g, {}).get("geneid", "")
        rows.append((g, ncbi, omim_gene.get(g, ""), gene_vocab.get(g, {}).get("uniprot", "")))
    return rows

def save_gene_id_table(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Gene", "NCBI_Entrez_GeneID", "OMIM_Gene_Number", "UniProt_ID_for_BLAST"])
        w.writerows(rows)
    return path

# #############################################################################
# B.  DATA MODELLING
# #############################################################################

def build_graph(dis_genes, omim_genes, seqs, homology_thr=0.30, use_blast=False):   # B1
    G = nx.Graph()
    def add_disease(node, genes):
        G.add_node(node, kind="disease")
        for gs, w in genes.items():
            G.add_node(gs, kind="gene")
            if G.has_edge(node, gs):
                G[node][gs]["weight"] = max(G[node][gs]["weight"], w)
            else:
                G.add_edge(node, gs, weight=w)
    for did, genes in dis_genes.items(): add_disease(did, genes)
    for did, genes in omim_genes.items(): add_disease(did, genes)
    present = {g: seqs[g] for g in seqs if g in G}
    homo = 0
    edges = blast_homology_edges(present, homology_thr) if use_blast else None
    if edges is not None:
        print("    A5: using REAL blastp (normalized bit-score)")
        for a, b, s in edges:
            G.add_edge(a, b, weight=s); homo += 1
    else:
        if use_blast: print("    A5: BLAST+ not found -> falling back to k-mer proxy")
        gs = list(present)
        for i in range(len(gs)):
            for j in range(i+1, len(gs)):
                s = seq_similarity(present[gs[i]], present[gs[j]])
                if s >= homology_thr:
                    G.add_edge(gs[i], gs[j], weight=s); homo += 1
    return G, homo

def precompute(G):
    nbrs, wts, nbrset = {}, {}, {}
    for n in G.nodes():
        ns = list(G.neighbors(n))
        nbrs[n]   = ns
        wts[n]    = np.array([G[n][m]["weight"] for m in ns], dtype=float)
        nbrset[n] = set(ns)
    return nbrs, wts, nbrset

def one_walk(start, length, p, q, nbrs, wts, nbrset, rng):                # B2
    walk = [start]
    if not nbrs[start]: return walk
    walk.append(rng.choice(nbrs[start], p=wts[start]/wts[start].sum()))
    while len(walk) < length:
        cur, prev = walk[-1], walk[-2]
        ns = nbrs[cur]
        if not ns: break
        w = wts[cur].copy()
        for idx, x in enumerate(ns):
            if x == prev:               w[idx] /= p
            elif x not in nbrset[prev]: w[idx] /= q
        walk.append(rng.choice(ns, p=w/w.sum()))
    return [str(x) for x in walk]

def node2vec_embed(G, dim=128, walk_len=40, num_walks=10, p=1.0, q=0.5,   # B2,B3
                   window=5, seed=42, workers=4):
    nbrs, wts, nbrset = precompute(G)
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    walks = []
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for n in nodes:
            walks.append(one_walk(n, walk_len, p, q, nbrs, wts, nbrset, rng))
    return Word2Vec(walks, vector_size=dim, window=window, min_count=0, sg=1,
                    workers=workers, seed=seed, epochs=5)

# #############################################################################
# C.  DATA EVALUATION
# #############################################################################

def gene_set(node, dis_genes, omim_genes):
    if node in dis_genes:  return dis_genes[node]
    if node in omim_genes: return omim_genes[node]
    return {}

def jaccard(a, b):                                                        # C1
    A, B = set(a), set(b)
    return len(A & B)/len(A | B) if (A | B) else 0.0

def weighted_cosine(a, b, gene_deg, N):                                   # C1
    keys = set(a) | set(b)
    def vec(d):
        return {g: d.get(g, 0.0) * math.log((N+1)/(gene_deg.get(g, 0)+1)) for g in keys}
    va, vb = vec(a), vec(b)
    dot = sum(va[g]*vb[g] for g in keys)
    na  = math.sqrt(sum(v*v for v in va.values()))
    nb  = math.sqrt(sum(v*v for v in vb.values()))
    return dot/(na*nb) if na and nb else 0.0

def n2v_cosine(model, n1, n2):                                           # C2
    if n1 not in model.wv or n2 not in model.wv: return None
    v1, v2 = model.wv[n1], model.wv[n2]
    c = float(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    return (c + 1.0)/2.0

def calibrate(model, raw, disease_nodes, k=400, seed=1):                  # C3
    rng = random.Random(seed); sample = []
    for _ in range(k):
        a, b = rng.choice(disease_nodes), rng.choice(disease_nodes)
        if a == b: continue
        s = n2v_cosine(model, a, b)
        if s is not None: sample.append(s)
    if not sample: return raw
    return sum(1 for s in sample if s <= raw)/len(sample)

def band(x):
    if x < 0.1:  return "~0   (unrelated / independent)"
    if x < 0.4:  return "~0.2 (not really similar)"
    if x < 0.75: return "~0.6 (somewhat / very similar)"
    return "~1   (highly / maximally similar)"

# ---- C4  Validation against the MeSH disease hierarchy (#16) -----------------
def pair_shared_categories(n1, n2, did2tree):
    return sorted(did2tree.get(n1, set()) & did2tree.get(n2, set()))

def ontology_auroc(model, disease_nodes, did2tree, k=600, seed=7):
    """Do node2vec scores agree with the MeSH ontology? Sample disease pairs,
    label 'similar' if they share a top MeSH category, score with node2vec cosine,
    and return AUROC (>0.5 means the embedding agrees with the ontology)."""
    cands = [d for d in disease_nodes if d in model.wv and d in did2tree]
    if len(cands) < 5: return None, 0, 0
    rng = random.Random(seed); y, s = [], []
    for _ in range(k):
        a, b = rng.choice(cands), rng.choice(cands)
        if a == b: continue
        sc = n2v_cosine(model, a, b)
        if sc is None: continue
        y.append(1 if (did2tree[a] & did2tree[b]) else 0); s.append(sc)
    if len(set(y)) < 2: return None, sum(y), len(y)
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s), sum(y), len(y)

# #############################################################################
# MODE 1:  run   (A -> B -> C similarity pipeline)
# #############################################################################
def cmd_run(args):
    if args.fast:
        args.dim, args.walk_len, args.num_walks = 64, 15, 3
    print("[A] Data processing ...")
    dis_genes, name2id, id2name, omim2mesh, gene_deg, gene2id = load_ctd(args.ctd)
    omim_genes, omim_name = load_omim(args.omim)
    seqs = load_fasta(args.fasta)
    gene_vocab = load_gene_vocab(args.genes)
    omim_gene  = load_omim_gene_numbers(args.omim_gene)
    did2tree   = load_disease_vocab(args.disvocab)          # #16
    N = len(dis_genes)
    print(f"    CTD {N} diseases | OMIM {len(omim_genes)} nodes | NCBI {len(seqs)} seqs")
    print(f"    A6 id-map: {len(gene2id)} NCBI ids | {len(gene_vocab)} UniProt | {len(omim_gene)} OMIM gene #")
    print(f"    C4 vocab : {len(did2tree)} diseases with MeSH categories (#16)")

    print("[B] Data modelling ...")
    G, homo = build_graph(dis_genes, omim_genes, seqs, use_blast=args.blast)
    print(f"    graph nodes={G.number_of_nodes()} edges={G.number_of_edges()} homology={homo}")
    model = node2vec_embed(G, dim=args.dim, walk_len=args.walk_len,
                           num_walks=args.num_walks, p=args.p, q=args.q)
    disease_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "disease"]

    print("[C] Data evaluation ...")
    pairs = [(args.d1, args.d2)] if args.d1 and args.d2 else [
        ("Breast Neoplasms", "Pancreatic Neoplasms"),
    ]
    for q1, q2 in pairs:
        n1, nm1 = resolve(q1, name2id, id2name, omim2mesh, omim_name, dis_genes, G)
        n2, nm2 = resolve(q2, name2id, id2name, omim2mesh, omim_name, dis_genes, G)
        print("\n" + "="*70)
        print(f"D1 = {q1!r} -> {nm1} [{n1}]")
        print(f"D2 = {q2!r} -> {nm2} [{n2}]")
        if not n1 or not n2:
            print("  could not resolve one disease"); continue
        g1, g2 = gene_set(n1, dis_genes, omim_genes), gene_set(n2, dis_genes, omim_genes)
        shared = sorted(set(g1) & set(g2))
        print(f"  genes(D1)={len(g1)} genes(D2)={len(g2)} shared={len(shared)}")
        print(f"  shared: {', '.join(shared[:20])}{' ...' if len(shared)>20 else ''}")
        rows = gene_id_table(shared, gene2id, gene_vocab, omim_gene)          # A6
        print(f"  [A6] gene ID table ({len(rows)} genes):")
        print("       {:<9}{:<10}{:<12}{}".format("Gene", "NCBI", "OMIM", "UniProt"))
        for g, ncbi, om, uni in rows:
            print("       {:<9}{:<10}{:<12}{}".format(g, ncbi, om or "-", uni or "-"))
        jac = jaccard(g1, g2)
        wco = weighted_cosine(g1, g2, gene_deg, N)
        raw = n2v_cosine(model, n1, n2)
        cal = calibrate(model, raw, disease_nodes) if raw is not None else None
        print(f"  [C1 baseline]  Jaccard         = {jac:.3f}")
        print(f"  [C1 baseline]  weighted cosine = {wco:.3f}")
        if raw is not None:
            print(f"  [C2 ML]        node2vec (raw)  = {raw:.3f}")
            print(f"  [C3 ML]        score (calib)   = {cal:.3f}  -> {band(cal)}")
        else:
            print("  [C2 ML] node not in embedding")
        # C4 : per-pair check against the MeSH hierarchy (#16)
        if did2tree:
            sc = pair_shared_categories(n1, n2, did2tree)
            verdict = f"share MeSH {sc} -> ontology: RELATED" if sc else "no shared MeSH branch -> ontology: unrelated"
            print(f"  [C4 validate]  {verdict}")

    # C4 : corpus-level validation (does node2vec agree with the ontology?)
    if did2tree:
        auroc, npos, ntot = ontology_auroc(model, disease_nodes, did2tree)
        if auroc is not None:
            print(f"\n[C4] Validation vs MeSH hierarchy (#16): AUROC = {auroc:.3f} "
                  f"over {ntot} random pairs ({npos} ontology-similar). >0.5 = agrees.")

# #############################################################################
# MODE 2:  fetch  (NCBI E-utilities -> real protein sequences for BLAST)
# #############################################################################
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
def _eutil(endpoint, params, api_key):
    import urllib.request, urllib.parse
    if api_key: params["api_key"] = api_key
    url = BASE + endpoint + "?" + urllib.parse.urlencode(params)
    return urllib.request.urlopen(url, timeout=30).read().decode()

def _fetch_protein_fasta(symbol, api_key):
    import json
    term = f"{symbol}[gene] AND Homo sapiens[orgn] AND refseq[filter]"
    j = json.loads(_eutil("esearch.fcgi",
        {"db": "protein", "term": term, "retmax": 1, "retmode": "json"}, api_key))
    ids = j.get("esearchresult", {}).get("idlist", [])
    if not ids: return None
    fasta = _eutil("efetch.fcgi",
        {"db": "protein", "id": ids[0], "rettype": "fasta", "retmode": "text"}, api_key)
    seq = "".join(l.strip() for l in fasta.splitlines() if l and not l.startswith(">"))
    return seq or None

def cmd_fetch(args):
    dis_genes, name2id, id2name, _, _, gene2id = load_ctd(args.ctd)
    if args.genes:
        genes = {g: gene2id.get(g, "") for g in args.genes}
    else:
        d1 = args.d1 or "Breast Neoplasms"
        d2 = args.d2 or "Pancreatic Neoplasms"
        n1 = resolve_name_only(d1, dis_genes, name2id, id2name)
        n2 = resolve_name_only(d2, dis_genes, name2id, id2name)
        genes = {}
        for n in (n1, n2):
            if n:
                for g in dis_genes[n]: genes[g] = gene2id.get(g, "")
        print(f"Genes from {id2name.get(n1)} + {id2name.get(n2)}: {len(genes)}")
    genes = dict(list(genes.items())[:args.max_genes])
    delay = 0.11 if args.api_key else 0.34
    written = 0
    with open(DEF_FASTA, "w", encoding="utf-8") as fh:
        fh.write("; Real human protein sequences fetched from NCBI E-utilities\n")
        for i, (sym, gid) in enumerate(genes.items(), 1):
            try:
                seq = _fetch_protein_fasta(sym, args.api_key)
            except Exception as e:
                print(f"  [{i}/{len(genes)}] {sym}: ERROR {e}"); seq = None
            if seq:
                fh.write(f">{sym}|{gid}\n{seq}\n"); written += 1
                print(f"  [{i}/{len(genes)}] {sym}: {len(seq)} aa")
            else:
                print(f"  [{i}/{len(genes)}] {sym}: no sequence")
            time.sleep(delay)
    print(f"\nWrote {written} sequences -> {DEF_FASTA}")

# #############################################################################
# MODE 3:  viz  (result figures)
# #############################################################################
PAIRS = [("Breast Neoplasms", "Pancreatic Neoplasms")]

def cmd_viz(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGDIR, exist_ok=True)

    dis_genes, name2id, id2name, omim2mesh, gene_deg, gene2id = load_ctd(DEF_CTD)
    omim_genes, omim_name = load_omim(DEF_OMIM)
    seqs = load_fasta(DEF_FASTA)
    N = len(dis_genes)
    G, _ = build_graph(dis_genes, omim_genes, seqs)
    made = []

    d1, d2 = "MESH:D001943", "MESH:D010190"
    g1, g2 = dis_genes[d1], dis_genes[d2]
    shared = sorted(set(g1) & set(g2), key=lambda g: -(g1[g] + g2[g]))[:18]
    fig, ax = plt.subplots(figsize=(9, 8))
    yc = np.linspace(0.05, 0.95, len(shared))
    for y, gene in zip(yc, shared):
        ax.plot([0.12, 0.5], [0.5, y], color="#b9c0c9", lw=1, zorder=1)
        ax.plot([0.5, 0.88], [y, 0.5], color="#b9c0c9", lw=1, zorder=1)
        ax.scatter(0.5, y, s=420, color="#8ecae6", edgecolor="#2a6f97", zorder=2)
        ax.text(0.5, y, gene, ha="center", va="center", fontsize=7.5, zorder=3)
    for x, did, col in [(0.12, d1, "#e07a5f"), (0.88, d2, "#81b29a")]:
        ax.scatter(x, 0.5, s=2600, color=col, edgecolor="black", zorder=4)
        ax.text(x, 0.5, id2name[did].split(",")[0].replace(" ", "\n"),
                ha="center", va="center", fontsize=8, weight="bold", zorder=5)
    ax.set_title(f"Shared-gene network ({len(set(g1)&set(g2))} shared; top 18 shown)")
    ax.axis("off")
    p = os.path.join(FIGDIR, "fig1_shared_network.png"); fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); made.append(p)

    model = None
    if not args.skip_ml:
        nw, wl, dim = (3, 15, 64) if args.fast else (10, 40, 128)
        model = node2vec_embed(G, dim=dim, walk_len=wl, num_walks=nw)
    disease_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "disease"]

    labels, jac, wco, n2v = [], [], [], []
    for a, b in PAIRS:
        na = resolve_name_only(a, dis_genes, name2id, id2name)
        nb = resolve_name_only(b, dis_genes, name2id, id2name)
        if not na or not nb: continue
        labels.append(f"{a.split()[0]}\nvs\n{b.split()[0]}")
        jac.append(jaccard(dis_genes[na], dis_genes[nb]))
        wco.append(weighted_cosine(dis_genes[na], dis_genes[nb], gene_deg, N))
        if model is not None:
            c = n2v_cosine(model, na, nb)
            n2v.append(calibrate(model, c, disease_nodes) if c else 0)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(0, jac[0], 0.5, color="#cbd5e1", label="Jaccard (baseline)")
    ax.bar(1, wco[0], 0.5, color="#94a3b8", label="Weighted cosine (baseline)")
    if n2v: ax.bar(2, n2v[0], 0.5, color="#2a6f97", label="node2vec (calibrated)")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Jaccard", "Weighted\ncosine", "node2vec"])
    ax.set_ylim(0, 1); ax.set_ylabel("similarity  [0-1]")
    ax.set_title("Breast vs Pancreatic: baselines vs node2vec")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    p = os.path.join(FIGDIR, "fig2_method_compare.png"); fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); made.append(p)

    if model is not None:
        from sklearn.decomposition import PCA
        highlight = ["MESH:D001943", "MESH:D010190"]
        dn = [n for n in disease_nodes if n in model.wv]
        sample = random.Random(0).sample(dn, min(400, len(dn)))
        for h in highlight:
            if h in model.wv and h not in sample: sample.append(h)
        xy = PCA(n_components=2).fit_transform(np.array([model.wv[n] for n in sample]))
        fig, ax = plt.subplots(figsize=(8.5, 7))
        ax.scatter(xy[:,0], xy[:,1], s=12, color="#cbd5e1", alpha=0.6)
        for k, (h, col) in enumerate(zip(highlight, ["#e07a5f", "#81b29a"])):
            if h in sample:
                i = sample.index(h); dy = 0.10 if k % 2 == 0 else -0.10
                va = "bottom" if k % 2 == 0 else "top"
                ax.scatter(xy[i,0], xy[i,1], s=120, color=col, edgecolor="black", zorder=3)
                ax.text(xy[i,0], xy[i,1]+dy, id2name.get(h, h).split(",")[0],
                        fontsize=9, weight="bold", ha="center", va=va, zorder=4)
        ax.set_title("node2vec disease embeddings (PCA to 2D)")
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.grid(alpha=0.3)
        p = os.path.join(FIGDIR, "fig3_embedding_map.png"); fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); made.append(p)

        rng = random.Random(1)
        sims = [s for _ in range(1500)
                for s in [n2v_cosine(model, rng.choice(dn), rng.choice(dn))] if s is not None]
        pair = n2v_cosine(model, d1, d2)
        cal  = sum(1 for s in sims if s <= pair)/len(sims)
        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.hist(sims, bins=40, color="#cbd5e1", edgecolor="white")
        ax.axvline(pair, color="#e07a5f", lw=2.5,
                   label=f"Breast-Pancreatic raw={pair:.2f}\ncalibrated={cal:.2f}")
        ax.set_title("Calibration: pair score vs 1500 random disease pairs")
        ax.set_xlabel("raw node2vec similarity [0-1]"); ax.set_ylabel("count of random pairs")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        p = os.path.join(FIGDIR, "fig4_calibration.png"); fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); made.append(p)

    print("Saved:")
    for p in made: print("   ", p)

# #############################################################################
# ENTRY POINT
# #############################################################################
def main():
    ap = argparse.ArgumentParser(description="Project-1 disease similarity (single-file).")
    sub = ap.add_subparsers(dest="mode", required=True)

    pr = sub.add_parser("run", help="compute similarity(D1, D2)")
    pr.add_argument("d1", nargs="?"); pr.add_argument("d2", nargs="?")
    pr.add_argument("--ctd", default=DEF_CTD); pr.add_argument("--omim", default=DEF_OMIM)
    pr.add_argument("--fasta", default=DEF_FASTA); pr.add_argument("--genes", default=DEF_GENES)
    pr.add_argument("--omim-gene", default=DEF_OMIMGENE)
    pr.add_argument("--disvocab", default=DEF_DISVOCAB)
    pr.add_argument("--dim", type=int, default=128); pr.add_argument("--walk-len", type=int, default=40)
    pr.add_argument("--num-walks", type=int, default=10)
    pr.add_argument("--p", type=float, default=1.0); pr.add_argument("--q", type=float, default=0.5)
    pr.add_argument("--fast", action="store_true")
    pr.add_argument("--blast", action="store_true", help="use REAL blastp for A5 (needs NCBI BLAST+)")
    pr.set_defaults(func=cmd_run)

    pf = sub.add_parser("fetch", help="collect protein sequences from NCBI (for BLAST)")
    pf.add_argument("d1", nargs="?"); pf.add_argument("d2", nargs="?")
    pf.add_argument("--genes", nargs="*", help="explicit gene symbols instead of diseases")
    pf.add_argument("--ctd", default=DEF_CTD)
    pf.add_argument("--api-key", default=os.environ.get("NCBI_API_KEY"))
    pf.add_argument("--max-genes", type=int, default=60)
    pf.set_defaults(func=cmd_fetch)

    pv = sub.add_parser("viz", help="generate result figures")
    pv.add_argument("--skip-ml", action="store_true"); pv.add_argument("--fast", action="store_true")
    pv.set_defaults(func=cmd_viz)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
