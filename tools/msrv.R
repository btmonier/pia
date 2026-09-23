# read the DESCRIPTION file
desc <- read.dcf("DESCRIPTION")

if (!"SystemRequirements" %in% colnames(desc)) {
    fmt <- c(
        "`SystemRequirements` not found in `DESCRIPTION`.",
        "Please specify `SystemRequirements: Cargo (Rust's package manager), rustc`"
    )
    stop(paste(fmt, collapse = "\n"))
}

# extract system requirements
sysreqs <- desc[, "SystemRequirements"]

# check that cargo and rustc is found
if (!grepl("cargo", sysreqs, ignore.case = TRUE)) {
    stop("You must specify `Cargo (Rust's package manager)` in your `SystemRequirements`")
}

if (!grepl("rustc", sysreqs, ignore.case = TRUE)) {
    stop("You must specify `Cargo (Rust's package manager), rustc` in your `SystemRequirements`")
}

# split into parts
parts <- strsplit(sysreqs, ", ")[[1]]

# identify which is the rustc
rustc_ver <- parts[grepl("rustc", parts)]

# perform checks for the presence of rustc and cargo on the OS
no_cargo_msg <- c(
    "----------------------- [CARGO NOT FOUND]--------------------------",
    "The 'cargo' command was not found on the PATH. Please install Cargo",
    "from: https://www.rust-lang.org/tools/install",
    "",
    "Alternatively, you may install Cargo from your OS package manager:",
    " - Debian/Ubuntu: apt-get install cargo",
    " - Fedora/CentOS: dnf install cargo",
    " - macOS: brew install rust",
    "-------------------------------------------------------------------"
)

no_rustc_msg <- c(
    "----------------------- [RUST NOT FOUND]---------------------------",
    "The 'rustc' compiler was not found on the PATH. Please install",
    paste(rustc_ver, "or higher from:"),
    "https://www.rust-lang.org/tools/install",
    "",
    "Alternatively, you may install Rust from your OS package manager:",
    " - Debian/Ubuntu: apt-get install rustc",
    " - Fedora/CentOS: dnf install rustc",
    " - macOS: brew install rust",
    "-------------------------------------------------------------------"
)

# Add {user}/.cargo/bin to path before checking. On Windows `HOME` is routinely
# redirected to the Documents folder (often OneDrive-backed), so `USERPROFILE`
# has to be consulted to locate a rustup installation.
cargo_roots <- Sys.getenv(c("CARGO_HOME", "USERPROFILE", "HOME"))
cargo_roots <- cargo_roots[nzchar(cargo_roots)]

cargo_bin <- c(
    file.path(cargo_roots[names(cargo_roots) == "CARGO_HOME"], "bin"),
    file.path(cargo_roots[names(cargo_roots) != "CARGO_HOME"], ".cargo", "bin")
)
cargo_bin <- unique(cargo_bin[dir.exists(cargo_bin)])

# Prepend, because `PATH` may have been inherited from a process that predates
# the Rust installation and so cannot be relied on to contain Cargo.
Sys.setenv(
    PATH = paste(
        c(cargo_bin, Sys.getenv("PATH")),
        collapse = .Platform[["path.sep"]]
    )
)

# check for rustc installation
rustc_version <- tryCatch(
    system("rustc --version", intern = TRUE),
    error = function(e) {
        stop(paste(no_rustc_msg, collapse = "\n"))
    }
)

# check for cargo installation
cargo_version <- tryCatch(
    system("cargo --version", intern = TRUE),
    error = function(e) {
        stop(paste(no_cargo_msg, collapse = "\n"))
    }
)

# helper function to extract versions
extract_semver <- function(ver) {
    if (grepl("\\d+\\.\\d+(\\.\\d+)?", ver)) {
        sub(".*?(\\d+\\.\\d+(\\.\\d+)?).*", "\\1", ver)
    } else {
        NA
    }
}

# get the MSRV
msrv <- extract_semver(rustc_ver)

# extract current version
current_rust_version <- extract_semver(rustc_version)

# perform check
if (!is.na(msrv)) {
    # -1 when current version is later
    # 0 when they are the same
    # 1 when MSRV is newer than current
    is_msrv <- utils::compareVersion(msrv, current_rust_version)
    if (is_msrv == 1) {
        fmt <- paste0(
            "\n------------------ [UNSUPPORTED RUST VERSION]------------------\n",
            "- Minimum supported Rust version is %s.\n",
            "- Installed Rust version is %s.\n",
            "---------------------------------------------------------------"
        )
        stop(sprintf(fmt, msrv, current_rust_version))
    }
}

# print the versions
versions_fmt <- "Using %s\nUsing %s"
message(sprintf(versions_fmt, cargo_version, rustc_version))
