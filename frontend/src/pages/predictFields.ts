export interface FieldConfig {
  name: string;
  label: string;
  type: "number" | "text";
  default: string | number;
}

// Mirrors LoanApplication in app/main.py — required fields first.
export const REQUIRED_FIELDS: FieldConfig[] = [
  { name: "ID", label: "Application ID", type: "number", default: 1 },
  { name: "year", label: "Year", type: "number", default: 2023 },
  { name: "loan_amount", label: "Loan amount", type: "number", default: 250000 },
  { name: "property_value", label: "Property value", type: "number", default: 320000 },
  { name: "income", label: "Monthly income", type: "number", default: 6000 },
  { name: "Credit_Score", label: "Credit score", type: "number", default: 720 },
];

// Optional in the backend (each already has a server-side default) — shown
// under "advanced" so the form is usable without touching all 28 fields.
export const OPTIONAL_FIELDS: FieldConfig[] = [
  { name: "loan_limit", label: "Loan limit", type: "text", default: "cf" },
  { name: "Gender", label: "Gender", type: "text", default: "Joint" },
  { name: "approv_in_adv", label: "Approved in advance", type: "text", default: "pre" },
  { name: "loan_type", label: "Loan type", type: "text", default: "type1" },
  { name: "loan_purpose", label: "Loan purpose", type: "text", default: "p3" },
  { name: "Credit_Worthiness", label: "Credit worthiness", type: "text", default: "l1" },
  { name: "open_credit", label: "Open credit", type: "text", default: "nopc" },
  { name: "business_or_commercial", label: "Business or commercial", type: "text", default: "nob/c" },
  { name: "rate_of_interest", label: "Rate of interest", type: "number", default: 4.5 },
  { name: "Interest_rate_spread", label: "Interest rate spread", type: "number", default: 0.9998 },
  { name: "Upfront_charges", label: "Upfront charges", type: "number", default: 5120.0 },
  { name: "term", label: "Term (months)", type: "number", default: 360 },
  { name: "Neg_ammortization", label: "Negative amortization", type: "text", default: "not_neg" },
  { name: "interest_only", label: "Interest only", type: "text", default: "not_int" },
  { name: "lump_sum_payment", label: "Lump sum payment", type: "text", default: "not_lpsm" },
  { name: "construction_type", label: "Construction type", type: "text", default: "sb" },
  { name: "occupancy_type", label: "Occupancy type", type: "text", default: "pr" },
  { name: "Secured_by", label: "Secured by", type: "text", default: "home" },
  { name: "total_units", label: "Total units", type: "text", default: "1U" },
  { name: "credit_type", label: "Credit type", type: "text", default: "EXP" },
  { name: "co_applicant_credit_type", label: "Co-applicant credit type", type: "text", default: "EXP" },
  { name: "age", label: "Age bracket", type: "text", default: "55-64" },
  { name: "submission_of_application", label: "Submission of application", type: "text", default: "to_inst" },
  { name: "LTV", label: "LTV", type: "number", default: 79.10958904 },
  { name: "Region", label: "Region", type: "text", default: "North" },
  { name: "Security_Type", label: "Security type", type: "text", default: "direct" },
  { name: "dtir1", label: "DTI ratio", type: "number", default: 44.0 },
];
