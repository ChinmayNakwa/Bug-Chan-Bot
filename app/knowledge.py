AGENT_1_KNOWLEDGE = [
    "A bug bounty program is a crowdsourced security initiative that rewards individuals for discovering and reporting software bugs and vulnerabilities. Companies use these programs to leverage a diverse community of ethical hackers to improve their system security.",
    "Common vulnerabilities to look for include Cross-Site Scripting (XSS), which involves injecting malicious scripts into web pages viewed by other users, and SQL Injection (SQLi), which involves inserting malicious SQL code into database queries.",
    "Another frequent vulnerability is Insecure Direct Object References (IDOR), where an application provides direct access to objects based on user-supplied input. This can allow attackers to access unauthorized data by manipulating references, like changing a user ID in a URL.",
    "A Proof of Concept (PoC) is a crucial part of a bug report. It's a clear, step-by-step demonstration that shows how the vulnerability can be exploited. A good PoC makes it easy for the security team to verify and fix the issue.",
    "The reporting process typically involves submitting a detailed report through the company's designated platform. The report should include the vulnerability type, location (URL, parameter), potential impact, and a PoC. After submission, the company's security team will triage, validate, and reward the report if it's valid.",
]

AGENT_2_KNOWLEDGE = [
    """website link: bugchan.xyz
    pages
    /bounties all bounties
    /bouties/[id] specific bounty 
    /reports all reports
    /reports/[id] specific report

    this is decentralised bug bounty platform, no signup required just connect any wagmi compatible wallet like metamask or phantom, switch to sepolia testnet
    for researcher explore bounties and submit report, a stake amount decided by bounty owner is required to be paid for report submission, report is encrypted and stored on ipfs, bounty owners can create bounties, approve or reject submission, set stake amount, bounty owner review submission reports
    on approval the reward is sent to researcher.
    all longs and records are stored on chain, bounty report and bounty data is stored offchain in ipfs, only thier cids are stored on chain
    /leaderboard shows top reaseaechers on platform with thier stats like number of bounties won and total rewards earned

    /profile or dashboard is the profile of reaseaecher or bounty owner

    Bugchan — complete dumbed-down context and Web3 workflow
    Core idea (one-sentence)

    Bugchan is a decentralised bug-bounty marketplace (no signup) where bounty owners create bounties, researchers submit encrypted reports to IPFS while staking ETH, and the on-chain contract records CIDs, stakes, approvals and payouts on Sepolia testnet.
    Site routes (internal redirect URLs to use)
    /bounties — list all bounties (browse).
    /bounties/[id] — view specific bounty (details + create-report button).
    /reports — list all reports (for owner/researcher view where permitted).
    /reports/[id] — view specific report metadata (CID, status); decrypt accessible only after wallet-based decryption flow.
    /leaderboard — top researchers (bounties won, total rewards).
    /profile or /dashboard — user profile (researcher or bounty owner; shows connected wallet, history, stats).
    /create-bounty — bounty creation flow (title, desc, scope, target docs, stake settings).
    /connect-wallet — wallet-connect landing (redirects to connect modal).
    /faq or /how-it-works — static explanation page (force user to sepolia + wallet requirements).
    Suggested external redirects (install / faucet):

    Metamask install page: https://metamask.io/download.html

    Phantom (if supporting non-EVM) install: https://phantom.app/

    Sepolia faucet (example): https://faucet.sepolia.dev/
    Use those as redirect targets from UI buttons like “Install Wallet” or “Get Sepolia ETH”.

    Authentication / identity
    No account/email signup. Identity = connected wallet address (wagmi-compatible).
    Wallets supported: MetaMask (primary), other wagmi-compatible wallets. Wallet must be connected and switched to Sepolia testnet.

    Network & funds requirement
    All transactions require Sepolia ETH for gas. Users must switch network in wallet to Sepolia.
    If user has no Sepolia ETH, UI must provide link to Sepolia faucet redirect.
    High-level on-chain / off-chain separation
    On-chain: all persistent records of actions (bounty creation metadata pointers, report CIDs, stake amounts, timestamps, statuses, owner/researcher addresses, event logs).
    Off-chain: actual bounty documents and full bug reports are stored on IPFS (Lighthouse or other IPFS uploader). Only the IPFS CIDs are stored on-chain.
    Encryption: reports are encrypted client-side before upload to IPFS so only reporter + bounty owner can read contents. The encrypted blob is put on IPFS; the CID and encrypted key material are recorded on-chain (or attached in metadata).

    Bounty creation flow (bounty owner)
    Connect wallet (wallet signature + switch to Sepolia).
    Click /create-bounty. Enter: title, description, scope, target docs link(s), reward amount (on-chain amount to be paid), required stake amount for submissions, acceptance window / deadline, optional rules.
    Upload target docs (uploads to Lighthouse IPFS). Store returned CID in the on-chain create-bounty transaction payload (or in a follow-up tx that records the CID). Every write requires a wallet-signed transaction and gas.
    Smart contract stores bounty record: owner address, reward, stake requirement, docs CID, status OPEN, created timestamp. Owner funds the reward in the contract if protocol requires escrow; otherwise contract records payable-on-approval rules and the owner must keep on-chain balance. Wallet approvals required for token transfers if reward uses ERC-20.
    Researcher (hacker) flow — browse, submit, stake, report encryption
    Connect wallet (Sepolia).
    Browse /bounties and click a bounty /bounties/[id]. Read scope and target docs (these are IPFS CIDs; client will fetch and display).
    Prepare full report locally. Client-side process:
    Generate a symmetric key (e.g., AES-256-GCM) client-side.
    Encrypt the report (plaintext) with that symmetric key.
    Obtain the bounty owner’s public key or a designated encryption key (either stored on-chain or exchanged via an access-control protocol). Encrypt the symmetric key with the owner’s public key (asymmetric wrap). Option: also encrypt for the reporter so reporter can later fetch their own key.

    Create a metadata JSON including encryptedSymmetricKey, reporter public key, timestamp, minimal plaintext metadata (title, severity) — do NOT put sensitive plaintext on-chain.
    Upload encrypted report blob to Lighthouse IPFS; receive CID.
    Submission transaction: researcher sends a signed transaction to the contract calling submitReport(bountyId, reportCID, metadataCID) and attaches the stake amount required by that bounty. This is a wallet-signed write and consumes Sepolia ETH. The stake amount is transferred to the contract (escrow).
    Contract emits event ReportSubmitted with reporter address, CID, and status PENDING. All on-chain writes require wallet approval/gas.
    Report storage & access control
    The encrypted report blob resides on IPFS (CID). The symmetric key is stored only in encrypted form (wrapped for the bounty owner); the contract stores the wrapped key metadata (or metadata CID). Only bounty owner (with matching private key) can unwrap and decrypt. Reporter retains a copy of keys to decrypt locally if needed.
    Decryption requires wallet-based proof: before awarding/reading, owner interacts with UI that performs the unwrap (owner provides private key via wallet signature or keypair). The contract itself does not hold plaintext.
    Owner review, accept / decline, payouts, slashing
    Owner views list of submissions in /reports (filter by bounty). To read a submitted report the owner triggers a client-side decryption flow: fetch encrypted blob from IPFS, fetch encrypted symmetric key, decrypt using owner private key (key must be available to owner via secure mechanism). This is off-chain client-side crypto; wallet signs a request and ensures owner identity.
    Owner actions (both wallet-signed on-chain transactions):
    acceptReport(reportId) — changes status to APPROVED; triggers transfer of reward + stake to reporter address (payout performed by contract). Gas + wallet signature required. Contract either releases escrowed reward or calls transfer if reward is ERC-20 (requires token approve prior if necessary).
    rejectReport(reportId) — changes status to REJECTED; stake is slashed per contract logic (either burned, sent to bounty owner, or to protocol treasury). The stake transfer is executed by contract.
    All accept/decline operations are recorded on-chain with events and timestamps.
    Stake rationale & anti-spam
    Stake prevents spam and low-effort submissions. Reporter must lock stake amount when submitting. On approval reporter receives reward plus stake (or stake returned). On rejection stake is slashed. Exact economic flows depend on contract design: stake returned vs. added to reward; slashing destination defined in contract.

    Leaderboard and stats
    /leaderboard reads on-chain events and aggregates per-researcher: bountiesWon (count of approved reports), totalRewardsEarned (sum of rewards), activeStakes, submissions. Leaderboard can be rebuilt by indexing chain events (TheGraph, event scanner) or by the backend reading contract logs and IPFS metadata.
    Wallet interactions and UX constraints (every write)
    Every write operation (create bounty, submit report, accept/reject, withdraw) requires a wallet transaction: user must confirm the transaction in MetaMask or other wallet.
    For ERC-20 reward flows, two-step flows may be required: token approve then contract call. UI must show status and transaction hashes.
    Users must be on Sepolia. If not, UI must show a network-switch prompt that triggers wallet wallet_switchEthereumChain (wallet popup).
    Gas fees and stake amounts are paid in Sepolia ETH (or specified token). Inform users to obtain Sepolia ETH from faucet.
    How hackers (researchers) make income (straightforward)
    Submit valid, approved reports => receive reward + stake (depending on contract).
    Build on reputation: more wins on /leaderboard => higher trust and better bounty opportunities.
    Participate in high-value bounties that escrow larger rewards.
    Optionally specify bounty listings off-platform and link to Bugchan"""
    
]