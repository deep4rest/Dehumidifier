using System.Security.Cryptography;
using System.Text.Json.Nodes;
using Dehumidifier.Simulation;
using Xunit;

namespace Dehumidifier.Tests;

public sealed class ModelContractTests
{
    private static string CopyModels()
    {
        var source=Path.Combine(AppContext.BaseDirectory,"Dehumidifier","models");
        if(!Directory.Exists(source)) source=Path.Combine(AppContext.BaseDirectory,"models");
        var target=Path.Combine(Path.GetTempPath(),"Dehumidifier.ModelContract.Tests",Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(target);
        foreach(var name in new[]{"model_contract.json","preprocessing.json","preprocessing_xf.json","preprocessing_main.json","xf.onnx","main.onnx"})
            File.Copy(Path.Combine(source,name),Path.Combine(target,name));
        return target;
    }

    [Theory]
    [InlineData("xf.onnx")]
    [InlineData("preprocessing.json")]
    [InlineData("preprocessing_xf.json")]
    public void MixedOrChangedFilesAreRejectedBeforeInference(string filename)
    {
        var root=CopyModels();
        try
        {
            File.AppendAllText(Path.Combine(root,filename)," ");
            var error=Assert.Throws<SimulationException>(()=>new OnnxRotor(root,"xf"));
            Assert.Equal("MODEL_CONTRACT_INVALID",error.Code);
            Assert.Contains("SHA256",error.Message);
        }
        finally {Directory.Delete(root,true);}
    }

    [Theory]
    [InlineData("columns")]
    [InlineData("clip")]
    [InlineData("zero_input_scale")]
    [InlineData("zero_output_scale")]
    [InlineData("log1p")]
    [InlineData("numeric_boundary")]
    public void ConsistentlyHashedButInvalidPreprocessingIsRejected(string mutation)
    {
        var root=CopyModels();
        try
        {
            var aggregate=JsonNode.Parse(File.ReadAllText(Path.Combine(root,"preprocessing.json")))!;
            var prep=aggregate["profiles"]!["xf"]!;
            switch(mutation)
            {
                case "columns":prep["input_columns"]![0]="错误列";break;
                case "clip":prep["clip"]=true;break;
                case "zero_input_scale":prep["input_scale"]![0]=0;break;
                case "zero_output_scale":prep["output_scale"]![0]=0;break;
                case "log1p":prep["log1p_output_columns"]=new JsonArray("处理出口温度");break;
                case "numeric_boundary":prep["input_scaling"]="old_float32_coefficients";break;
            }
            File.WriteAllText(Path.Combine(root,"preprocessing_xf.json"),prep.ToJsonString());
            File.WriteAllText(Path.Combine(root,"preprocessing.json"),aggregate.ToJsonString());
            var contract=JsonNode.Parse(File.ReadAllText(Path.Combine(root,"model_contract.json")))!;
            string Hash(string name)=>Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(Path.Combine(root,name)))).ToLowerInvariant();
            contract["preprocessing_sha256"]=Hash("preprocessing.json");
            var model=contract["models"]!.AsArray().Single(x=>x!["profile"]!.GetValue<string>()=="xf")!;
            model["preprocessing_profile_sha256"]=Hash("preprocessing_xf.json");
            File.WriteAllText(Path.Combine(root,"model_contract.json"),contract.ToJsonString());
            Assert.Equal("MODEL_CONTRACT_INVALID",Assert.Throws<SimulationException>(()=>new OnnxRotor(root,"xf")).Code);
        }
        finally {Directory.Delete(root,true);}
    }
}
