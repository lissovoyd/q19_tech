# Phase 3 — Model Comparison: Cluster Labels & Ops Report

Same 17 clusters, same 5 representative queries per cluster fed to each model. Differences reflect model language and verbosity, not different input data.

## Cost Summary

| | Claude Haiku 4.5 | Qwen2.5 7B (local) |
|---|---|---|
| Model | `claude-haiku-4-5-20251001` | `qwen2.5:7b-instruct-q4_K_M` |
| Calls | 18 (17 labelling + 1 report) | 18 (17 labelling + 1 report) |
| Tokens (report call) | 1282 | 817 |
| Cost (report call) | Actual: $0.0029 | Hypothetical: $0.0002 |

## Cluster Label Comparison

| # | Size | Claude Haiku 4.5 | Qwen2.5 7B (local) |
|---|---|---|---|
| 0 | 17 | Duplicate Charge Issues | Duplicate Charges |
| 1 | 27 | Card Payment Declined Issues | Card Payment Issues |
| 2 | 48 | Pending Payment Status Inquiries | Pending Payment Inquiry |
| 3 | 38 | Lost Or Stolen Card | Card Lost/Stolen |
| 4 | 19 | Failed Top-Up Issues | Top-Up Failure Reasons |
| 5 | 19 | Refund Request | Refund Assistance |
| 6 | 18 | Transfer Cancellation Requests | Cancel Transfer |
| 7 | 22 | Card Delivery Status Inquiry | Card Delivery Issues |
| 8 | 16 | Contactless Payment Troubleshooting | Contactless Troubleshooting |
| 9 | 20 | Update Personal Details ✓ | Update Personal Details ✓ |
| 10 | 37 | Transfer Completion Issues | Transfer Issues |
| 11 | 20 | Identity Verification Issues ✓ | Identity Verification Issues ✓ |
| 12 | 21 | PIN Change Process | PIN Change Procedures |
| 13 | 21 | Unrecognized Card Payment | Unrecognized Card Payments |
| 14 | 17 | PIN Unblocking And Reset | PIN Unblock Queries |
| 15 | 20 | Unexpected Fee On Statement | Statement Extra Fee Inquiry |
| 16 | 20 | Card Activation Assistance | Card Activation |

_✓ = identical label. Differences are mostly phrasing, not meaning._

## Description Comparison

| # | Claude Haiku 4.5 | Qwen2.5 7B (local) |
|---|---|---|
| 0 | **Duplicate Charge Issues**<br>Customers reporting being charged multiple times for a single transaction or purchase. | **Duplicate Charges**<br>Queries regarding repeated charges for the same transaction |
| 1 | **Card Payment Declined Issues**<br>Customers reporting problems with their credit or debit card payments being declined during transactions. | **Card Payment Issues**<br>Queries regarding declined card payments |
| 2 | **Pending Payment Status Inquiries**<br>Customers asking why their card payments remain in pending status and have not been processed. | **Pending Payment Inquiry**<br>Queries regarding the status of a payment that has not been processed. |
| 3 | **Lost Or Stolen Card**<br>Customers reporting that their payment card has been lost or stolen and need assistance. | **Card Lost/Stolen**<br>Queries related to missing or stolen cards |
| 4 | **Failed Top-Up Issues**<br>Customers reporting or inquiring about unsuccessful top-up transactions and seeking explanations for the failures. | **Top-Up Failure Reasons**<br>Queries regarding reasons for unsuccessful top-ups |
| 5 | **Refund Request**<br>Customers requesting to return items and receive refunds for their purchases. | **Refund Assistance**<br>Queries related to requesting and processing refunds |
| 6 | **Transfer Cancellation Requests**<br>Customers seeking assistance with canceling transfers they have initiated or completed. | **Cancel Transfer**<br>Queries regarding the cancellation of transfers |
| 7 | **Card Delivery Status Inquiry**<br>Customers inquiring about the status or delayed delivery of their new card orders. | **Card Delivery Issues**<br>Queries regarding delays or lack of receipt of new cards. |
| 8 | **Contactless Payment Troubleshooting**<br>Customers seeking help to diagnose and resolve issues with their contactless payment functionality. | **Contactless Troubleshooting**<br>Queries regarding issues with contactless technology and methods to resolve them. |
| 9 | **Update Personal Details**<br>Customers requesting to modify or update their personal information such as address or account details. | **Update Personal Details**<br>Queries related to modifying or changing personal information. |
| 10 | **Transfer Completion Issues**<br>Customers reporting problems with transfers that fail to process or complete successfully. | **Transfer Issues**<br>Queries regarding problems with transfers not going through. |
| 11 | **Identity Verification Issues**<br>Customers unable to complete or pass identity verification processes. | **Identity Verification Issues**<br>Queries related to problems verifying personal identity |
| 12 | **PIN Change Process**<br>Customers asking how to change their personal identification number and where they can do so. | **PIN Change Procedures**<br>Queries regarding methods and processes for changing a PIN. |
| 13 | **Unrecognized Card Payment**<br>Customers reporting unauthorized or unfamiliar transactions on their card statements that they do not recognize. | **Unrecognized Card Payments**<br>Queries regarding unrecognized transactions on credit or debit cards. |
| 14 | **PIN Unblocking And Reset**<br>Customers seeking help to unblock or reset their blocked PIN. | **PIN Unblock Queries**<br>Queries related to blocking and unblocking personal identification numbers. |
| 15 | **Unexpected Fee On Statement**<br>Customers inquiring about an unexplained or unexpected fee appearing on their account statement. | **Statement Extra Fee Inquiry**<br>Queries regarding unexpected fees in statements |
| 16 | **Card Activation Assistance**<br>Customers requesting help with activating their card or inquiring about the activation process. | **Card Activation**<br>Queries related to activating a card. |

## Ops Report Narrative Comparison

| Claude Haiku 4.5 | Qwen2.5 7B (local) |
|---|---|
| This week's 400 queries reveal a support landscape dominated by transaction and card management issues, with Pending Payment Inquiry (48 queries, 12.0%) significantly outpacing other clusters. Security and fraud concerns—including Card Lost/Stolen, Unrecognized Card Payments, and Duplicate Charges—collectively represent 76 queries (19.0%), indicating a notable customer trust and account safety dimension. Transfer-related issues (Transfer Issues, Cancel Transfer) account for 55 queries (13.7%), suggesting potential friction in a core banking function. | This week, the banking app handled 400 customer support queries across 17 issue clusters. The most significant clusters by volume are Pending Payment Inquiry and Card Lost/Stolen. These issues require urgent attention due to their high frequency. |

## Escalation Clusters Comparison

| Claude Haiku 4.5 | Qwen2.5 7B (local) |
|---|---|
| **Unrecognized Card Payments**<br>Fraud and unauthorized transaction reports require immediate investigation and potential account protection measures, regardless of query volume. Delayed response risks customer financial loss and regulatory compliance issues. | **Pending Payment Inquiry**<br>High query volume indicates potential system or process inefficiencies that need immediate resolution. |
| **Card Lost/Stolen**<br>Security-critical cluster requiring urgent card blocking and replacement to prevent unauthorized use. High volume (38 queries) combined with security sensitivity demands prioritized handling. | **Card Lost/Stolen**<br>Critical security and user experience issue requiring prompt action to prevent fraud and ensure customer safety |
| **Identity Verification Issues**<br>Account access and security bottleneck. Customers unable to verify identity may be locked out of critical functions, requiring expedited resolution to restore service. |  |
| **Pending Payment Inquiry**<br>Highest volume cluster (48 queries) suggests potential systemic delays in payment processing. Investigate root cause to prevent customer dissatisfaction and possible payment failures. |  |

## Recommendations Comparison

| | Claude Haiku 4.5 | Qwen2.5 7B (local) |
|---|---|---|
| 1 | Implement automated status notifications for pending payments to reduce inquiry volume; analyze the 48 Pending Payment Inquiry cases to identify if delays stem from a specific payment method, time window, or transaction type. | Implement real-time payment status updates to reduce pending inquiry queries. |
| 2 | Create a self-service fraud reporting and card blocking feature to handle Unrecognized Card Payments and Card Lost/Stolen cases without agent intervention, reducing response time for security-critical issues. | Enhance card tracking systems to minimize lost or stolen card incidents. |
| 3 | Develop a streamlined identity verification process (e.g., multi-factor authentication options, video verification) to reduce Identity Verification Issues and unblock customers faster. |  |
| 4 | Audit transfer processing logic to identify why Transfer Issues and Cancel Transfer collectively represent 13.7% of queries; prioritize fixes for the most common failure reasons. |  |
| 5 | Establish a knowledge base and chatbot for high-volume, low-complexity clusters (PIN Change Procedures, Card Activation, Update Personal Details) to deflect routine queries and free agent capacity for complex cases. |  |
