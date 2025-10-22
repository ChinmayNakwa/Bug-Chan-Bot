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

    /profile or dashboard is the profile of reaseaecher or bounty owner"""
]