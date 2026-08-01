import { useState, useEffect, useCallback, useRef } from "react";
import { useSearchParams } from "react-router-dom";
import { apiGet, apiPost, ApiError } from "../api/client";

interface Plan {
  id: string;
  label: string;
  monthly_quota: number;
  price_label: string;
  price_amount: number | null;
  description: string;
}
interface UsageResponse {
  plan: string;
  used: number;
  limit: number;
}
interface CheckoutResponse {
  authorization_url: string;
}
interface BillingPortalResponse {
  link: string;
}

export default function Plans() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [plans, setPlans] = useState<Plan[]>([]);
  const [currentPlan, setCurrentPlan] = useState<string | null>(null);
  const [switching, setSwitching] = useState<string | null>(null);
  const [openingPortal, setOpeningPortal] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Paystack's hosted checkout replaces the whole callback_url query string
  // with its own trxref/reference params on redirect back — it doesn't
  // preserve or merge any query params we pass in, so "did we just come
  // back from checkout" has to be detected from those, not a marker of
  // our own choosing.
  const [confirming, setConfirming] = useState(
    searchParams.has("reference") || searchParams.has("trxref")
  );
  const planBeforeCheckout = useRef<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [plansResp, usageResp] = await Promise.all([
        apiGet<{ plans: Plan[] }>("/v1/plans"),
        apiGet<UsageResponse>("/v1/usage"),
      ]);
      setPlans(plansResp.plans);
      setCurrentPlan(usageResp.plan);
      return usageResp.plan;
    } catch {
      // non-fatal — the page just won't render plan cards this tick
      return null;
    }
  }, []);

  useEffect(() => {
    load().then((plan) => {
      if (!confirming) planBeforeCheckout.current = plan;
    });
  }, [load, confirming]);

  // Paystack redirects the browser back here after a checkout attempt —
  // the webhook that actually confirms payment can land a couple seconds
  // after that redirect, so poll briefly rather than trusting the plan is
  // already updated on the very first render.
  useEffect(() => {
    if (!confirming) return;
    let attempts = 0;
    const interval = setInterval(async () => {
      attempts += 1;
      const plan = await load();
      if (plan !== planBeforeCheckout.current || attempts >= 8) {
        setConfirming(false);
        setSearchParams({}, { replace: true });
        clearInterval(interval);
      }
    }, 1500);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [confirming]);

  async function switchTo(planId: string) {
    setError(null);
    setSwitching(planId);
    try {
      await apiPost("/v1/tenant/plan", { plan: planId });
      setCurrentPlan(planId);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to switch plan");
    } finally {
      setSwitching(null);
    }
  }

  async function checkout(planId: string) {
    setError(null);
    setSwitching(planId);
    try {
      const resp = await apiPost<CheckoutResponse>("/v1/tenant/checkout", { plan: planId });
      window.location.href = resp.authorization_url;
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to start checkout");
      setSwitching(null);
    }
  }

  async function openBillingPortal() {
    setError(null);
    setOpeningPortal(true);
    try {
      const resp = await apiGet<BillingPortalResponse>("/v1/tenant/billing-portal");
      window.open(resp.link, "_blank", "noopener,noreferrer");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to open billing portal");
    } finally {
      setOpeningPortal(false);
    }
  }

  const hasActiveSubscription = currentPlan !== null && currentPlan !== "free";

  return (
    <div className="page">
      <h1>Subscription plan</h1>
      <p>Choose the tier that matches your monthly prediction volume. Paid tiers are billed through Paystack.</p>
      {confirming && <p className="plan-tag">Confirming payment with Paystack…</p>}
      {error && <p className="error">{error}</p>}
      {hasActiveSubscription && (
        <button type="button" onClick={openBillingPortal} disabled={openingPortal}>
          {openingPortal ? "Opening..." : "Manage billing"}
        </button>
      )}

      <div className="plan-grid">
        {plans.map((plan) => {
          const isCurrent = plan.id === currentPlan;
          const isCheckoutTier = plan.price_amount !== null && plan.price_amount > 0;
          return (
            <div key={plan.id} className={`plan-card${isCurrent ? " current" : ""}`}>
              {isCurrent && <span className="badge">Current plan</span>}
              <h2 className="plan-card-label">{plan.label}</h2>
              <div className="plan-card-price">{plan.price_label}</div>
              <div className="plan-card-quota">{plan.monthly_quota.toLocaleString()} predictions / mo</div>
              <p className="plan-card-desc">{plan.description}</p>
              {plan.price_amount === null ? (
                !isCurrent && <p className="plan-card-desc">Contact sales to enable this tier.</p>
              ) : (
                <button
                  type="button"
                  disabled={isCurrent || switching !== null}
                  onClick={() => (isCheckoutTier ? checkout(plan.id) : switchTo(plan.id))}
                >
                  {isCurrent
                    ? "Current plan"
                    : switching === plan.id
                      ? isCheckoutTier
                        ? "Redirecting..."
                        : "Switching..."
                      : "Switch to this plan"}
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
