cd(@__DIR__)
load_data("dataset.csv", dlm = ",")

nn1 = load_model("model.jld2")
#nn2 = load_model("model.h5")

save_model(nn1)
#save_model(nn2)
