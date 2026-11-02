# Persist every audit result (tables and posterior summaries) for the paper's figures and macros.
import pickle
pickle.dump(AUD, open("audit_results.pkl", "wb"))
print("saved audit_results.pkl:", sorted(AUD))
