
FPGA_USER = "petalinux"
FPGA_PASSWORD = "root"

PROJECT_CONFIG = {
    1: {
        "name": "Damage Conf",
        "output_file_damage": "./output_damage.csv",
        "output_file_no_damage": "./output_no_damage.csv",
        "config_file": "./config_vdfd.txt",
        "remote_output": "/home/petalinux/output_vdfd.csv",
        "remote_config": "/home/petalinux/config_vdfd.txt",
        "command": "vdfd",
        "description": """Version con Streams de distancias. Resolución configurable."""

    },
    2: {
        "name": "Damage V2",
        "output_file_damage": "./outputV2.csv",
        "output_file_no_damage": "./outputV2_non_damage.csv",
        "config_file": "./config_vdm.txt",
        "remote_output": "/home/petalinux/output_vdm.csv",
        "remote_config": "/home/petalinux/config_vdm.txt",
        "command": "vdm",
        "description": """Version con distancias en memoria. Resolución fija en 120x120."""
    },
    3: {
        "name": "Damage Dif",
        "output_file_damage": "outputDif.csv",
        "output_file_no_damage": "outputDif_non_damage.csv",
        "config_file": "config_vad.txt",
        "remote_output": "/home/petalinux/output_vad.csv",
        "remote_config": "/home/petalinux/config_vad.txt",
        "command": "vad",
        "description":"""Versión que calcula la diferencia con las señales de referencia. Resolución configurable"""
    }
}

RESOURCE_TABLE = {
    "krnl_conf": {
        "LUT": 90716, "FF": 48617, "BRAM": 29, "URAM": 48, "DSP": 151,
        "LUT%": "80.05%", "FF%": "21.31%", "BRAM%": "21.97%", "URAM%": "75.00%", "DSP%": "12.10%"
    },
    "krnl_lat": {
        "LUT": 85809, "FF": 47463, "BRAM": 50, "URAM": 55, "DSP": 503,
        "LUT%": "75.72%", "FF%": "20.80%", "BRAM%": "37.88%", "URAM%": "85.94%", "DSP%": "40.26%"
    },
    "krnl_dif": {
        "LUT": 91065, "FF": 51414, "BRAM": 45, "URAM": 52, "DSP": 515,
        "LUT%": "80.36%", "FF%": "22.53%", "BRAM%": "34.09%", "URAM%": "81.25%", "DSP%": "41.27%"
    }
}

