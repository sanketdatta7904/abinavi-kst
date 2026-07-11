#!/usr/bin/env Rscript
# Generates JSON reference fixtures by calling the REAL kstMatrix package.
# NEVER edits kstMatrix. Run from the package root:
#   Rscript tests/r_reference/generate_fixtures.R
suppressMessages(library(kstMatrix))
suppressMessages(library(jsonlite))

out_dir <- "tests/fixtures"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

write_fixture <- function(name, obj) {
  path <- file.path(out_dir, paste0(name, ".json"))
  writeLines(toJSON(obj, digits = 17, auto_unbox = TRUE, null = "null"), path)
  cat("wrote", path, "\n")
}

# Shared reference structure: xpl$space (9 states x 4 items).
xpl_space <- unclass(xpl$space)
storage.mode(xpl_space) <- "integer"

write_fixture("meta", list(
  r_version = as.character(getRversion()),
  kstMatrix_version = as.character(packageVersion("kstMatrix")),
  double_eps_tol = sqrt(.Machine$double.eps),
  xpl_space = xpl_space
))

# ---- kmassessbayesian cases -------------------------------------------------
# Prior used across bayesian cases (sums to 1 within tolerance):
bp <- c(0.02, 0.1, 0.07, 0.01, 0.4, 0.17, 0.07, 0.08, 0.08)
beta4 <- rep(0.2, 4)
eta4  <- rep(0.1, 4)

bayes_case <- function(probs, ks, beta, eta, question, response) {
  list(
    inputs = list(probs = probs, beta = beta, eta = eta,
                  question = question, response = response),
    r_output = kmassessbayesian(probs, ks, beta, eta, question, response)
  )
}

write_fixture("bayesian_B1_resp1", bayes_case(bp, xpl_space, beta4, eta4, 3, 1))
write_fixture("bayesian_B2_resp0", bayes_case(bp, xpl_space, beta4, eta4, 3, 0))
# B3: distinct per-item beta/eta
write_fixture("bayesian_B3_vecparams",
  bayes_case(bp, xpl_space, c(0.05, 0.1, 0.2, 0.3), c(0.02, 0.05, 0.1, 0.15), 2, 1))
# B4: non-uniform prior already covered by bp; add uniform prior variant
up <- rep(1/9, 9)
write_fixture("bayesian_B4_uniformprior",
  bayes_case(up, xpl_space, beta4, eta4, 4, 1))
# B5: every item as question, response=1
for (q in 1:4) {
  write_fixture(sprintf("bayesian_B5_q%d", q),
                bayes_case(bp, xpl_space, beta4, eta4, q, 1))
}

# Error/warning behavior captured as recorded messages, not values:
capture <- function(expr) {
  err <- NULL; warn <- NULL
  val <- withCallingHandlers(
    tryCatch(expr, error = function(e) { err <<- conditionMessage(e); NULL }),
    warning = function(w) { warn <<- c(warn, conditionMessage(w)); invokeRestart("muffleWarning") }
  )
  list(error = err, warnings = warn)
}

write_fixture("bayesian_err_nonbinary",
  capture(kmassessbayesian(up, matrix(c(0L,2L,1L,1L), 2, 2), c(0.1,0.1), c(0.1,0.1), 1, 1)))
write_fixture("bayesian_err_probs_sum",
  capture(kmassessbayesian(rep(0.5, 9), xpl_space, beta4, eta4, 1, 1)))
write_fixture("bayesian_err_beta_null",
  capture(kmassessbayesian(bp, xpl_space, NULL, eta4, 1, 1)))
write_fixture("bayesian_err_beta_len",
  capture(kmassessbayesian(bp, xpl_space, rep(0.1, 3), eta4, 1, 1)))
# beta+eta>1 is a WARNING, not an error — value still returned:
write_fixture("bayesian_warn_beta_eta",
  capture(kmassessbayesian(bp, xpl_space, rep(0.6,4), rep(0.6,4), 1, 1)))

# ---- kmassesshalfsplit cases ------------------------------------------------
# We cannot capture R's random tie-break reproducibly. Instead we record the
# CANDIDATE SET (items tied at min distance-to-0.5), computed the same way R does.
halfsplit_candidates <- function(probs, ks) {
  noi <- dim(ks)[2]
  ip <- sapply(1:noi, function(x) sum(probs[which(ks[, x] == 1)]))
  hd <- abs(ip - 0.5)
  which(hd == min(hd))
}

hs_case <- function(probs, ks) {
  list(
    inputs = list(probs = probs),
    item_prob = sapply(1:dim(ks)[2], function(x) sum(probs[which(ks[, x] == 1)])),
    candidates = halfsplit_candidates(probs, ks),
    # Deterministic only when candidates has length 1:
    deterministic = length(halfsplit_candidates(probs, ks)) == 1
  )
}

# H3: the documented example (returns 2, unique min under this prior).
write_fixture("halfsplit_H3_example",
  hs_case(c(0.02, 0.1, 0.07, 0.01, 0.4, 0.17, 0.07, 0.08, 0.08), xpl_space))
# H1: another prior producing a unique min.
write_fixture("halfsplit_H1_unique",
  hs_case(c(0.5, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05), xpl_space))
# H2: uniform prior — expect a tie (candidate set length > 1).
write_fixture("halfsplit_H2_tie", hs_case(rep(1/9, 9), xpl_space))

write_fixture("halfsplit_err_probs_sum",
  capture(kmassesshalfsplit(rep(0.5, 9), xpl_space)))
write_fixture("halfsplit_err_nonbinary",
  capture(kmassesshalfsplit(rep(1/2, 2), matrix(c(0L,2L,1L,1L), 2, 2))))

# ---- kmsassess / kmassess full loop ----------------------------------------
# Force a deterministic query sequence: all-correct and all-wrong responses
# make halfsplit ties irrelevant to the FINAL state (state is unequivocal),
# though the queried order may still vary on ties. We therefore record state
# and probs (order-invariant as a multiset) plus the queried sequence for
# inspection. Tests compare state exactly and probs as a sorted multiset.
sassess_case <- function(r, ks, threshold) {
  res <- kmsassess(r, ks, "halfsplit", "Bayesian", 0.1, 0.1, NULL, NULL, threshold)
  list(
    inputs = list(r = r, beta = 0.1, eta = 0.1, threshold = threshold),
    state = res$state,
    probs = res$probs,
    queried = res$queried
  )
}

# S1 all-correct (full mastery) and all-wrong (empty state):
write_fixture("sassess_S1_allcorrect", sassess_case(c(1,1,1,1), xpl_space, 0.55))
write_fixture("sassess_S1_allwrong",   sassess_case(c(0,0,0,0), xpl_space, 0.55))
# S4 uniform-prior construction is implicit in kmsassess; record nos for check:
write_fixture("sassess_S4_prior",
  list(nos = dim(xpl_space)[1], uniform = rep(1/dim(xpl_space)[1], dim(xpl_space)[1])))

# S2 probdev=TRUE returns a list of distributions:
res_pd <- kmsassess(c(1,1,1,1), xpl_space, "halfsplit", "Bayesian",
                    0.1, 0.1, NULL, NULL, 0.55, probdev = TRUE)
write_fixture("sassess_S2_probdev", list(
  n_steps = length(res_pd$probs),
  first = res_pd$probs[[1]],
  last = res_pd$probs[[length(res_pd$probs)]],
  queried = res_pd$queried
))

# S5 tie in max: capture whatever R's which(probs==max) yields on a crafted
# probability vector via kmassessbayesian alone (loop not needed):
p_tie <- rep(1/9, 9)
write_fixture("sassess_S5_whichmax", list(
  probs = p_tie,
  which_max = which(p_tie == max(p_tie))  # expect all 9 indices
))

# Error: threshold out of [0,1] (stop):
write_fixture("sassess_err_threshold",
  capture(kmsassess(c(1,1,1,1), xpl_space, "halfsplit", "Bayesian",
                    0.1, 0.1, NULL, NULL, 1.5)))
# Warning: threshold<=0.5 (warns, continues):
write_fixture("sassess_warn_threshold",
  capture(kmsassess(c(1,1,1,1), xpl_space, "halfsplit", "Bayesian",
                    0.1, 0.1, NULL, NULL, 0.4)))
# Warning: Bayesian with zeta set (warns, continues):
write_fixture("sassess_warn_zeta",
  capture(kmsassess(c(1,1,1,1), xpl_space, "halfsplit", "Bayesian",
                    0.1, 0.1, 2, 2, 0.55)))

# ---- Caveat 1: multi-winner state flattening (S-4, distinct tied rows) ------
# R builds state via result <- which(probs==max(probs)); as.integer(ks[result,]).
# When result has >1 index AND the tied rows differ, R flattens the submatrix
# COLUMN-MAJOR. These fixtures pin R's exact byte order so the Python mirror
# (reshape(-1, order="F")) is verified against ground truth, not assumed.
whichmax_state <- function(ks, probs) {
  storage.mode(ks) <- "integer"
  result <- which(probs == max(probs))
  list(
    probs = probs,
    which_max = result,
    n_winners = length(result),
    # Exactly kmassess.R:251-253:
    state = as.integer(ks[result, ])
  )
}

# Two DISTINCT rows tied for max -> exercises the column-major flatten:
ks_distinct <- rbind(c(1L,0L,0L), c(0L,1L,1L))
write_fixture("state_multiwinner_distinct",
  whichmax_state(ks_distinct, c(0.5, 0.5)))
# Three distinct rows tied:
ks_three <- rbind(c(1L,0L,0L,0L), c(0L,1L,0L,0L), c(0L,0L,1L,1L))
write_fixture("state_multiwinner_three",
  whichmax_state(ks_three, c(1/3, 1/3, 1/3)))
# Single winner (control):
write_fixture("state_singlewinner",
  whichmax_state(ks_three, c(0.6, 0.2, 0.2)))

# ---- Caveat 2: larger, real R structure end-to-end -------------------------
# A 6-item surmise relation, union-closed to an 18-state space by kmspace().
# Verifies kmsassess/kmassess equivalence beyond the 9x4 xpl$space.
big_basis <- rbind(
  c(1,0,0,0,0,0),  # a
  c(1,1,0,0,0,0),  # b <- a
  c(1,0,1,0,0,0),  # c <- a
  c(0,0,0,1,0,0),  # d
  c(0,0,0,1,1,0),  # e <- d
  c(1,1,1,0,0,1)   # f <- a,b,c
)
storage.mode(big_basis) <- "integer"
class(big_basis) <- c("kmbasis", "kmfamset", "matrix", "array")
big_space <- unclass(kmspace(big_basis))
storage.mode(big_space) <- "integer"

write_fixture("sassess_big_meta", list(
  space = big_space,
  nos = dim(big_space)[1],
  noi = dim(big_space)[2]
))
big_case <- function(r, threshold) {
  res <- kmsassess(r, big_space, "halfsplit", "Bayesian",
                   0.1, 0.1, NULL, NULL, threshold)
  list(
    inputs = list(r = r, beta = 0.1, eta = 0.1, threshold = threshold),
    state = res$state,
    probs = res$probs,
    queried = res$queried
  )
}
write_fixture("sassess_big_allcorrect", big_case(rep(1, 6), 0.55))
write_fixture("sassess_big_allwrong",   big_case(rep(0, 6), 0.55))
