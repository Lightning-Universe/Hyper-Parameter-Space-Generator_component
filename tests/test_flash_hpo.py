from flash_hpo import FlashHPO

def test_flash_hpo_component_text_classification():
    # Test the Flash HPO Component for text classification
    hpo_config = {
        "backbone": "prajjwal1/bert-tiny",
        "learning_rate": [0.00001, 0.01],
    }

    hpo = FlashHPO()
    hpo.run(hpo_config) 
    assert hpo.generated_runs, "Didn't generate runs..."
