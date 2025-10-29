from mistral_baseline import run


def run_baseline():
    # CZ
    print("Czech prompting")
    run(inp_path="datasets/CZ/test.json",
        out_path="predictions/zero_shot/predictions_cz.json",
        prompt="zero-shot")
    # VN
    print("Vietnamese prompting")
    run(inp_path="datasets/VNM/test.json",
        out_path="predictions/zero_shot/predictions_vnm.json",
        prompt="zero-shot")
    # RU
    print("Russian prompting")
    run(inp_path="datasets/RU/test.json",
        out_path="predictions/zero_shot/predictions_ru.json",
        prompt="zero-shot")
    # EN
    print("English prompting")
    run(inp_path="datasets/EN/test.json",
        out_path="predictions/zero_shot/predictions_en.json",
        prompt="zero-shot")
    # MULTI
    print("Multi prompting")
    run(inp_path="datasets/MULTI/test.json",
        out_path="predictions/zero_shot/predictions_multi.json",
        prompt="zero-shot")

    ####################################################

    # CZ
    print("Czech prompting")
    run(inp_path="datasets/CZ/test.json",
        out_path="predictions/few_shot/predictions_cz.json",
        prompt="few-shot")
    # VN
    print("Vietnamese prompting")
    run(inp_path="datasets/VNM/test.json",
        out_path="predictions/few_shot/predictions_vnm.json",
        prompt="few-shot")
    # RU
    print("Russian prompting")
    run(inp_path="datasets/RU/test.json",
        out_path="predictions/few_shot/predictions_ru.json",
        prompt="few-shot")
    # EN
    print("English prompting")
    run(inp_path="datasets/EN/test.json",
        out_path="predictions/few_shot/predictions_en.json",
        prompt="few-shot")
    # MULTI
    print("Multi prompting")
    run(inp_path="datasets/MULTI/test.json",
        out_path="predictions/few_shot/predictions_multi.json",
        prompt="few-shot")

    ####################################################

    # CZ
    print("Czech prompting")
    run(inp_path="datasets/CZ/test.json",
        out_path="predictions/structure-based/predictions_cz.json",
        prompt="structure-based")
    # VN
    print("Vietnamese prompting")
    run(inp_path="datasets/VNM/test.json",
        out_path="predictions/structure-based/predictions_vnm.json",
        prompt="structure-based")
    # RU
    print("Russian prompting")
    run(inp_path="datasets/RU/test.json",
        out_path="predictions/structure-based/predictions_ru.json",
        prompt="structure-based")
    # EN
    print("English prompting")
    run(inp_path="datasets/EN/test.json",
        out_path="predictions/structure-based/predictions_en.json",
        prompt="structure-based")
    # MULTI
    print("Multi prompting")
    run(inp_path="datasets/MULTI/test.json",
        out_path="predictions/structure-based/predictions_multi.json",
        prompt="structure-based")

    ####################################################

    # CZ
    print("Czech prompting")
    run(inp_path="datasets/CZ/test.json",
        out_path="predictions/instruction-based/predictions_cz.json",
        prompt="instruction-based")
    # VN
    print("Vietnamese prompting")
    run(inp_path="datasets/VNM/test.json",
        out_path="predictions/instruction-based/predictions_vnm.json",
        prompt="instruction-based")
    # RU
    print("Russian prompting")
    run(inp_path="datasets/RU/test.json",
        out_path="predictions/instruction-based/predictions_ru.json",
        prompt="instruction-based")
    # EN
    print("English prompting")
    run(inp_path="datasets/EN/test.json",
        out_path="predictions/instruction-based/predictions_en.json",
        prompt="instruction-based")
    # MULTI
    print("Multi prompting")
    run(inp_path="datasets/MULTI/test.json",
        out_path="predictions/instruction-based/predictions_multi.json",
        prompt="instruction-based")




def main():
    print("Running baseline...")
    run_baseline()


if __name__ == "__main__":
    main()
