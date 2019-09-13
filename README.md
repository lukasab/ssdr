# ssdr
ssdr is a Python script with OpenCV for identifying and interpret seven-segment displays. 

### Input:

![image](Last_capture/Last_capture.png) 

### Output:

![image](https://user-images.githubusercontent.com/16520334/64829559-3493d580-d5a3-11e9-829a-124d2850aac7.png)

![image](https://user-images.githubusercontent.com/16520334/64829598-5d1bcf80-d5a3-11e9-9b38-49626a6abdcd.png)

## Installation

Clone this project with:

```bash
git clone https://github.com/lukasab/ssdr.git
```

Install requirements using [pipenv](https://github.com/pypa/pipenv) or use:
```bash
pip install -r requirements.txt
```

## Usage
To run the example shown here with all images that the script generates you can run:

```bash
python main.py -i Last_capture/Last_capture.png -d True
```

Press any key to continue the script.

If you want to use it with your image just change `Last_capture/Last_capture.png` for your image path.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
The content of this project is licensed under the [MIT license](LICENSE).